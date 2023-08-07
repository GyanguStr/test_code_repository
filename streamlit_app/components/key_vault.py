import os
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class FetchKey:
    """class that fetch key from Azure key vault"""

    load_dotenv(verbose=True)

    secret_value: str
    tenant_id: str = (
        os.getenv("KEY_TENANT_ID") if os.getenv("KEY_TENANT_ID") else ""
    )
    client_id: str = (
        os.getenv("KEY_CLIENT_ID") if os.getenv("KEY_CLIENT_ID") else ""
    )
    client_secret: str = (
        os.getenv("KEY_CLIENT_SECRET") if os.getenv("KEY_CLIENT_SECRET") else ""
    )
    KVName: str = (
        os.getenv("KEY_VAULT_NAME") if os.getenv("KEY_VAULT_NAME") else "wpb-key-vault"
    )
    KVUri: str = field(init=False)

    def __post_init__(self):
        self.KVUri = f"https://{self.KVName}.vault.azure.net"

    def retrieve_secret(self) -> str:
        credential = ClientSecretCredential(
            self.tenant_id,
            self.client_id,
            self.client_secret
        )
        client = SecretClient(vault_url=self.KVUri, credential=credential)
        retrieved_secret = client.get_secret(self.secret_value)
        return retrieved_secret.value



