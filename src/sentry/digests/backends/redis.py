import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import rb
from rb.clients import LocalClient
from redis.exceptions import ResponseError

from sentry.digests import Record, ScheduleEntry
from sentry.digests.backends.base import Backend, InvalidState
from sentry.utils.locking.backends.redis import RedisLockBackend
from sentry.utils.locking.lock import Lock
from sentry.utils.locking.manager import LockManager
from sentry.utils.redis import check_cluster_versions, get_cluster_from_options, load_redis_script
from sentry.utils.versioning import Version

logger = logging.getLogger("sentry.digests")

script = load_redis_script("digests/digests.lua")


class RedisBackend(Backend):
    """
    Implements the digest backend API, backed by Redis.

    Each timeline is modeled as a sorted set, and also maintains a separate key
    that contains the last time the digest was processed (used for scheduling.)

    .. code::

        redis:6379> ZREVRANGEBYSCORE "d:t:mail:p:1" inf -inf WITHSCORES
        1) "433be20b807c4cd49a132de69c0f6c55"
        2) "1444847625"
        3) "0f9d5fe4b5b3400fab85d9a841aa8467"
        4) "1444847625"
        ...

    The timeline contains references to several records, which are stored
    separately, encoded using the codec provided to the backend:

    .. code::

        redis:6379> GET "d:t:mail:p:1:r:433be20b807c4cd49a132de69c0f6c55"
        [ binary content ]

    When the timeline is ready to be digested, the timeline set is renamed,
    creating a digest set (in this case the key would be ``d:t:mail:p:1:d``),
    that represents a snapshot of the timeline contents at that point in time.
    (If the digest set already exists, the timeline contents are instead
    unioned into the digest set and then the timeline is cleared.) This allows
    new records to be added to the timeline that will be processed after the
    next scheduling interval without the risk of data loss due to race
    conditions between the record addition and digest generation and delivery.

    Schedules are modeled as two sorted sets -- one for ``waiting`` items, and
    one for ``ready`` items. Items in the ``waiting`` set are scored by the
    time at which they should be transitioned to the ``ready`` set.  Items in
    the ``ready`` set are scored by the time at which they were scheduled to be
    added to the ``ready`` set. Iterating each set from oldest to newest yields
    the highest priority items for action (moving from the ``waiting`` to
    ``ready`` set, or delivering a digest for the ``waiting`` and ``ready``
    set, respectively.)

    .. code::

        redis:6379> ZREVRANGEBYSCORE "d:s:w" inf -inf WITHSCORES
        1) "mail:p:1"
        2) "1444847638"

    """

    def __init__(self, **options: Any) -> None:
        cluster, options = get_cluster_from_options("SENTRY_DIGESTS_OPTIONS", options)
        assert isinstance(cluster, rb.Cluster)
        self.cluster = cluster
        self.locks = LockManager(RedisLockBackend(self.cluster))

        self.namespace = options.pop("namespace", "d")

        # Sets the time-to-live (in seconds) for records, timelines, and
        # digests. This can (and should) be a relatively high value, since
        # timelines, digests, and records should all be deleted after they have
        # been processed -- this is mainly to ensure stale data doesn't hang
        # around too long in the case of a configuration error. This should be
        # larger than the maximum scheduling delay to ensure data is not evicted
        # too early.
        self.ttl = options.pop("ttl", 60 * 60)

        super().__init__(**options)

    def validate(self) -> None:
        logger.debug("Validating Redis version...")
        check_cluster_versions(self.cluster, Version((2, 8, 9)), label="Digests")

    def _get_connection(self, key: str) -> LocalClient:
        return self.cluster.get_local_client_for_key(f"{self.namespace}:t:{key}")

    def _get_timeline_lock(self, key: str, duration: int) -> Lock:
        lock_key = f"{self.namespace}:t:{key}"
        return self.locks.get(
            lock_key, duration=duration, routing_key=lock_key, name="digest_timeline_lock"
        )

    def add(
        self,
        key: str,
        record: Record,
        increment_delay: int | None = None,
        maximum_delay: int | None = None,
        timestamp: float | None = None,
    ) -> bool:
        if timestamp is None:
            timestamp = time.time()

        if increment_delay is None:
            increment_delay = self.increment_delay

        if maximum_delay is None:
            maximum_delay = self.maximum_delay

        # Redis returns "true" and "false" as "1" and "None", so we just cast
        # them back to the appropriate boolean here.
        return bool(
            script(
                [key],
                [
                    "ADD",
                    self.namespace,
                    self.ttl,
                    timestamp,
                    key,
                    record.key,
                    self.codec.encode(record.value),
                    record.timestamp,  # TODO: check type
                    increment_delay,
                    maximum_delay,
                    self.capacity if self.capacity else -1,
                    self.truncation_chance,
                ],
                self._get_connection(key),
            )
        )

    def __schedule_partition(
        self, host: int, deadline: float, timestamp: float
    ) -> Iterable[tuple[bytes, float]]:
        return script(
            ["-"],
            ["SCHEDULE", self.namespace, self.ttl, timestamp, deadline],
            self.cluster.get_local_client(host),
        )

    def schedule(self, deadline: float, timestamp: float | None = None) -> Iterable[ScheduleEntry]:
        if timestamp is None:
            timestamp = time.time()

        for host in self.cluster.hosts:
            try:
                for key, timestamp in self.__schedule_partition(host, deadline, timestamp):
                    yield ScheduleEntry(key.decode("utf-8"), float(timestamp))
            except Exception as error:
                logger.exception(
                    "Failed to perform scheduling for partition %s due to error: %s",
                    host,
                    error,
                )

    def __maintenance_partition(self, host: int, deadline: float, timestamp: float) -> Any:
        return script(
            ["-"],
            ["MAINTENANCE", self.namespace, self.ttl, timestamp, deadline],
            self.cluster.get_local_client(host),
        )

    def maintenance(self, deadline: float, timestamp: float | None = None) -> None:
        if timestamp is None:
            timestamp = time.time()

        for host in self.cluster.hosts:
            try:
                self.__maintenance_partition(host, deadline, timestamp)
            except Exception as error:
                logger.exception(
                    "Failed to perform maintenance on digest partition %s due to error: %s",
                    host,
                    error,
                )

    @contextmanager
    def digest(
        self, key: str, minimum_delay: int | None = None, timestamp: float | None = None
    ) -> Any:
        if minimum_delay is None:
            minimum_delay = self.minimum_delay

        if timestamp is None:
            timestamp = time.time()

        connection = self._get_connection(key)
        with self._get_timeline_lock(key, duration=30).acquire():
            try:
                response = script(
                    [key],
                    [
                        "DIGEST_OPEN",
                        self.namespace,
                        self.ttl,
                        timestamp,
                        key,
                        self.capacity if self.capacity else -1,
                    ],
                    connection,
                )
            except ResponseError as e:
                if "err(invalid_state):" in str(e):
                    raise InvalidState("Timeline is not in the ready state.") from e
                else:
                    raise

            records = [
                Record(
                    key.decode(),
                    self.codec.decode(value) if value is not None else None,
                    float(timestamp),
                )
                for key, value, timestamp in response
            ]

            # If the record value is `None`, this means the record data was
            # missing (it was presumably evicted by Redis) so we don't need to
            # return it here.
            filtered_records = [record for record in records if record.value is not None]
            if len(records) != len(filtered_records):
                logger.warning(
                    "Filtered out missing records when fetching digest",
                    extra={
                        "key": key,
                        "record_count": len(records),
                        "filtered_record_count": len(filtered_records),
                    },
                )
            yield filtered_records

            script(
                [key],
                ["DIGEST_CLOSE", self.namespace, self.ttl, timestamp, key, minimum_delay]
                + [record.key for record in records],
                connection,
            )

    def delete(self, key: str, timestamp: float | None = None) -> None:
        if timestamp is None:
            timestamp = time.time()

        connection = self._get_connection(key)
        with self._get_timeline_lock(key, duration=30).acquire():
            script([key], ["DELETE", self.namespace, self.ttl, timestamp, key], connection)
