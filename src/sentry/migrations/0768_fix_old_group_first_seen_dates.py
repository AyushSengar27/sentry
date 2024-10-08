# Generated by Django 5.1.1 on 2024-09-24 20:28

from datetime import datetime, timezone

from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor

from sentry.new_migrations.migrations import CheckedMigration

OLD_FIRST_SEEN_CUTOFF = datetime(2000, 1, 1, tzinfo=timezone.utc)


def update_old_first_seen_dates(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    Group = apps.get_model("sentry", "Group")

    for group in Group.objects.filter(first_seen__lt=OLD_FIRST_SEEN_CUTOFF):
        group.first_seen = group.active_at
        group.save(update_fields=["first_seen"])


class Migration(CheckedMigration):
    # This flag is used to mark that a migration shouldn't be automatically run in production.
    # This should only be used for operations where it's safe to run the migration after your
    # code has deployed. So this should not be used for most operations that alter the schema
    # of a table.
    # Here are some things that make sense to mark as post deployment:
    # - Large data migrations. Typically we want these to be run manually so that they can be
    #   monitored and not block the deploy for a long period of time while they run.
    # - Adding indexes to large tables. Since this can take a long time, we'd generally prefer to
    #   run this outside deployments so that we don't block them. Note that while adding an index
    #   is a schema change, it's completely safe to run the operation after the code has deployed.
    # Once deployed, run these manually via: https://develop.sentry.dev/database-migrations/#migration-deployment

    is_post_deployment = True

    dependencies = [
        ("sentry", "0767_add_selected_aggregate_to_dashboards_widget_query"),
    ]

    operations = [
        migrations.RunPython(
            update_old_first_seen_dates,
            migrations.RunPython.noop,
            hints={"tables": ["sentry_groupedmessage"]},
        ),
    ]
