import logging
import azure
import os



from azure.health.deidentification import DeidentificationClient
from azure.identity import DefaultAzureCredential
from azure.health.deidentification.models import (
        DeidentificationResult,
        DeidentificationContent,
        OperationType,
        DocumentDataType,
    )


logging.basicConfig(level=logging.DEBUG)

#os.environ['AZURE_CLIENT_ID'] = ''
#os.environ['AZURE_TENANT_ID'] = ''
#os.environ['AZURE_CLIENT_SECRET'] = ''
#print(os.environ.get("AZURE_CLIENT_ID"))
#print(os.environ.get("AZURE_TENANT_ID"))
#print(os.environ.get("AZURE_CLIENT_SECRET"))

if __name__ == '__main__':
    # credential should auto-load from cli login with azd auth login --use-device-code

    credential = DefaultAzureCredential()
    client = DeidentificationClient(endpoint='b8gyfeayd5gnfjhc.api.eus001.deid.azure.com', credential=credential)

    body = DeidentificationContent(input_text="Hello, my name is John Smith.")

    result: DeidentificationResult = client.deidentify(body)
    print(result)
