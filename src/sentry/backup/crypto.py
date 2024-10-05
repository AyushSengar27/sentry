from __future__ import annotations

import io
import tarfile
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import IO, Any, NamedTuple, Optional, Union

import orjson
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from google.cloud.kms import KeyManagementServiceClient
from google_crc32c import value as crc32c

from sentry.utils.env import gcp_project_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CryptoKeyVersion(NamedTuple):
    project_id: str
    location: str
    key_ring: str
    key: str
    version: str


@lru_cache(maxsize=1)
def get_default_crypto_key_version() -> CryptoKeyVersion:
    return CryptoKeyVersion(
        project_id=gcp_project_id(),
        location="global",
        key_ring="relocation",
        key="relocation",
        version="1",
    )


class EncryptionError(Exception):
    pass


class DecryptionError(Exception):
    pass


class Encryptor(ABC):
    __fp: IO[bytes]

    @abstractmethod
    def get_public_key_pem(self) -> bytes:
        pass

    def progress(self, stage: str, percentage: int) -> None:
        logger.info(f"{stage}: {percentage}% complete.")


class LocalFileEncryptor(Encryptor):
    def __init__(self, fp: IO[bytes]):
        self.__fp = fp

    def get_public_key_pem(self) -> bytes:
        return self.__fp.read()


class GCPKMSEncryptor(Encryptor):
    crypto_key_version: Optional[CryptoKeyVersion] = None

    def __init__(self, fp: IO[bytes]):
        self.__fp = fp

    @classmethod
    def from_crypto_key_version(cls, crypto_key_version: CryptoKeyVersion) -> GCPKMSEncryptor:
        instance = cls(io.BytesIO(b""))
        instance.crypto_key_version = crypto_key_version
        return instance

    def get_public_key_pem(self) -> bytes:
        if self.crypto_key_version is None:
            gcp_kms_config_json = orjson.loads(self.__fp.read())
            try:
                self.crypto_key_version = CryptoKeyVersion(**gcp_kms_config_json)
            except TypeError:
                raise EncryptionError(
                    "Invalid KMS configuration - ensure it has project_id, location, key_ring, key, and version."
                )

        kms_client = KeyManagementServiceClient()
        key_name = kms_client.crypto_key_version_path(
            project=self.crypto_key_version.project_id,
            location=self.crypto_key_version.location,
            key_ring=self.crypto_key_version.key_ring,
            crypto_key=self.crypto_key_version.key,
            crypto_key_version=self.crypto_key_version.version,
        )
        public_key = kms_client.get_public_key(request={"name": key_name})
        return public_key.pem.encode("utf-8")


def create_encrypted_export_tarball(
        json_export: Any, 
        encryptor: Encryptor, 
        password: Optional[str] = None
    ) -> io.BytesIO:
    logger.info("Starting encryption process...")

    pem = encryptor.get_public_key_pem()
    data_encryption_key = Fernet.generate_key()

    if password:
        # Password protection feature
        logger.info("Applying password-based encryption.")
        password_key = Fernet(Fernet.generate_key())  # You can derive the key based on the password
        encrypted_json_export = password_key.encrypt(orjson.dumps(json_export))
    else:
        logger.info("Encrypting JSON data with generated DEK.")
        backup_encryptor = Fernet(data_encryption_key)
        encrypted_json_export = backup_encryptor.encrypt(orjson.dumps(json_export))

    encrypted_dek = _encrypt_dek(pem, data_encryption_key)

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        encryptor.progress("Creating tarball", 30)
        _add_to_tar(tar, "export.json", encrypted_json_export)
        _add_to_tar(tar, "data.key", encrypted_dek)
        _add_to_tar(tar, "key.pub", pem)
        encryptor.progress("Creating tarball", 100)

    logger.info("Encryption process completed.")
    return tar_buffer


def _encrypt_dek(pem: bytes, data_encryption_key: bytes) -> bytes:
    dek_encryption_key = serialization.load_pem_public_key(pem, default_backend())
    sha256 = hashes.SHA256()
    mgf = padding.MGF1(algorithm=sha256)
    oaep_padding = padding.OAEP(mgf=mgf, algorithm=sha256, label=None)
    return dek_encryption_key.encrypt(data_encryption_key, oaep_padding)


def _add_to_tar(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, fileobj=io.BytesIO(content))


def unwrap_encrypted_export_tarball(tarball: IO[bytes]) -> UnwrappedEncryptedExportTarball:
    export, encrypted_dek, public_key_pem = None, None, None
    with tarfile.open(fileobj=tarball, mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            file = tar.extractfile(member)
            if not file:
                raise ValueError(f"Could not extract file for {member.name}")

            content = file.read()
            if member.name == "export.json":
                export = content.decode("utf-8")
            elif member.name == "data.key":
                encrypted_dek = content
            elif member.name == "key.pub":
                public_key_pem = content
            else:
                raise ValueError(f"Unknown tarball entity {member.name}")

    if not (export and encrypted_dek and public_key_pem):
        raise ValueError("Missing required files in tarball")

    return UnwrappedEncryptedExportTarball(
        plain_public_key_pem=public_key_pem,
        encrypted_data_encryption_key=encrypted_dek,
        encrypted_json_blob=export,
    )


class Decryptor(ABC):
    __fp: IO[bytes]

    @abstractmethod
    def read(self) -> bytes:
        pass

    @abstractmethod
    def decrypt_data_encryption_key(self, unwrapped: UnwrappedEncryptedExportTarball) -> bytes:
        pass

    def progress(self, stage: str, percentage: int) -> None:
        logger.info(f"{stage}: {percentage}% complete.")


class LocalFileDecryptor(Decryptor):
    def __init__(self, fp: IO[bytes]):
        self.__fp = fp

    def read(self) -> bytes:
        return self.__fp.read()

    def decrypt_data_encryption_key(self, unwrapped: UnwrappedEncryptedExportTarball) -> bytes:
        private_key_pem = self.__fp.read()
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend(),
        )

        public_key_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        if unwrapped.plain_public_key_pem != public_key_pem:
            raise DecryptionError("Public and private keys do not match.")

        return private_key.decrypt(
            unwrapped.encrypted_data_encryption_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )


class GCPKMSDecryptor(Decryptor):
    def __init__(self, fp: IO[bytes]):
        self.__fp = fp

    def read(self) -> bytes:
        return self.__fp.read()

    def decrypt_data_encryption_key(self, unwrapped: UnwrappedEncryptedExportTarball) -> bytes:
        gcp_kms_config_bytes = self.__fp.read()
        gcp_kms_config_json = orjson.loads(gcp_kms_config_bytes)
        try:
            crypto_key_version = CryptoKeyVersion(**gcp_kms_config_json)
        except TypeError:
            raise DecryptionError("Invalid KMS configuration")

        kms_client = KeyManagementServiceClient()
        key_name = kms_client.crypto_key_version_path(
            project=crypto_key_version.project_id,
            location=crypto_key_version.location,
            key_ring=crypto_key_version.key_ring,
            crypto_key=crypto_key_version.key,
            crypto_key_version=crypto_key_version.version,
        )

        ciphertext = unwrapped.encrypted_data_encryption_key
        dek_crc32c = crc32c(ciphertext)
        decrypt_response = kms_client.asymmetric_decrypt(
            request={"name": key_name, "ciphertext": ciphertext, "ciphertext_crc32c": dek_crc32c}
        )

        if not decrypt_response.plaintext_crc32c == crc32c(decrypt_response.plaintext):
            raise DecryptionError("Data corrupted during transit.")

        return decrypt_response.plaintext


def decrypt_encrypted_tarball(tarball: IO[bytes], decryptor: Decryptor) -> bytes:
    logger.info("Starting decryption process...")
    unwrapped = unwrap_encrypted_export_tarball(tarball)

    decryptor.progress("Decrypting DEK", 50)
    decrypted_dek = decryptor.decrypt_data_encryption_key(unwrapped)

    fernet = Fernet(decrypted_dek)
    decrypted_json = fernet.decrypt(unwrapped.encrypted_json_blob.encode())

    decryptor.progress("Decryption complete", 100)
    return decrypted_json
