import os
from pathlib import Path
import json

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


SMARTSHEET_TOKEN = os.getenv("SMARTSHEET_TOKEN")
SMARTSHEET_SHEET_ID = os.getenv("SMARTSHEET_SHEET_ID")
VIN_COLUMN_TITLE = os.getenv("SMARTSHEET_VIN_COLUMN_TITLE", "Complete VIN")

# Optional: path to your corporate root CA bundle (PEM)
CORP_CA_BUNDLE = os.getenv("CORP_CA_BUNDLE")

# If CORP_CA_BUNDLE is set, use it; otherwise, you can choose to disable
# verification for Smartsheet calls by setting DISABLE_SM_SSL_VERIFY=true in .env.
disable_verify = os.getenv("DISABLE_SM_SSL_VERIFY", "false").lower() == "true"
if CORP_CA_BUNDLE:
    VERIFY_ARG = CORP_CA_BUNDLE
elif disable_verify:
    VERIFY_ARG = False
else:
    VERIFY_ARG = True

SM_API_BASE = "https://api.smartsheet.com/2.0"

if not SMARTSHEET_TOKEN or not SMARTSHEET_SHEET_ID:
    raise RuntimeError(
        "SMARTSHEET_TOKEN and SMARTSHEET_SHEET_ID must be set in the environment or .env file."
    )

SM_HEADERS = {
    "Authorization": f"Bearer {SMARTSHEET_TOKEN}",
    "Content-Type": "application/json",
}


# For debugging: limit how many full attachment JSON blobs we print
DEBUG_ATTACHMENTS_DUMP_LIMIT = 5
DEBUG_ATTACHMENTS_DUMPED = 0


def get_sheet():
    url = f"{SM_API_BASE}/sheets/{SMARTSHEET_SHEET_ID}"
    response = requests.get(url, headers=SM_HEADERS, timeout=30, verify=VERIFY_ARG)
    response.raise_for_status()
    return response.json()


def build_column_map(columns):
    return {col["id"]: col["title"] for col in columns}

def extract_vin_from_row(row, col_map, vin_column_title=VIN_COLUMN_TITLE):
    for cell in row.get("cells", []):
        col_id = cell.get("columnId")
        if not col_id:
            continue
        title = col_map.get(col_id, "")
        if title.strip() != vin_column_title:
            continue
        if "value" in cell and cell["value"] is not None:
            return str(cell["value"]).strip()
        if "displayValue" in cell and cell["displayValue"] is not None:
            return str(cell["displayValue"]).strip()
    return None


def get_row_attachments(row_id):
    url = f"{SM_API_BASE}/sheets/{SMARTSHEET_SHEET_ID}/rows/{row_id}/attachments"
    try:
        response = requests.get(url, headers=SM_HEADERS, timeout=30, verify=VERIFY_ARG)
        response.raise_for_status()
    except requests.RequestException as exc:
        # Handle HTTP errors, timeouts, and connection issues without killing the script
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"Error fetching attachments for row {row_id} (HTTP {status}): {exc}")
        return []

    # Smartsheet list responses use the "data" key for collections
    return response.json().get("data", [])


def download_attachment(attachment, fallback_name="attachment"):
    """Download a Smartsheet FILE attachment via the official download endpoint.

    For attachmentType == "FILE", Smartsheet does not include a direct file URL
    on the row-attachments object. Instead we must call
    `/sheets/{sheetId}/attachments/{attachmentId}/download` and follow the
    redirect to the actual file.
    """

    global DEBUG_ATTACHMENTS_DUMPED

    attachment_id = attachment.get("id")
    filename = attachment.get("name") or fallback_name
    att_type = attachment.get("attachmentType") or attachment.get("type")

    if att_type and att_type != "FILE":
        # Caller already filters non-FILE types, but keep this guard.
        print(
            f"Attachment {attachment_id} is type {att_type}, not FILE; "
            "skipping this attachment."
        )
        return None, None

    download_url = f"{SM_API_BASE}/sheets/{SMARTSHEET_SHEET_ID}/attachments/{attachment_id}/download"

    try:
        file_resp = requests.get(
            download_url,
            allow_redirects=True,
            timeout=60,
            headers=SM_HEADERS,
            verify=VERIFY_ARG,
        )
        file_resp.raise_for_status()
        content = file_resp.content
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(
            f"Error downloading attachment {attachment_id} via download endpoint "
            f"(HTTP {status}): {exc}. Skipping this attachment."
        )

        if DEBUG_ATTACHMENTS_DUMPED < DEBUG_ATTACHMENTS_DUMP_LIMIT:
            DEBUG_ATTACHMENTS_DUMPED += 1
            print("Raw attachment object:")
            try:
                print(json.dumps(attachment, indent=2))
            except TypeError:
                print(repr(attachment))

        return None, None

    return content, filename


def main():
    script_dir = Path(__file__).resolve().parent
    base_output_dir = script_dir / "VIN_PICs"
    base_output_dir.mkdir(exist_ok=True)

    sheet = get_sheet()
    rows = sheet.get("rows", [])
    columns = sheet.get("columns", [])
    col_map = build_column_map(columns)

    print(f"Found {len(rows)} rows in Smartsheet sheet {SMARTSHEET_SHEET_ID}.")

    for row in rows:
        row_id = row.get("id")
        row_number = row.get("rowNumber")
        vin = extract_vin_from_row(row, col_map)
        if not vin:
            print(
                f"RowId {row_id} (RowNumber {row_number}): no VIN in column "
                f"'{VIN_COLUMN_TITLE}', skipping."
            )
            continue

        vin_dir = base_output_dir / vin
        vin_dir.mkdir(exist_ok=True)

        attachments = get_row_attachments(row_id)
        if not attachments:
            print(
                f"RowId {row_id} (RowNumber {row_number}, VIN {vin}): no attachments."
            )
            continue

        for attachment in attachments:
            # Skip non-file attachments (e.g. links) that can't be downloaded as files
            att_type = attachment.get("attachmentType") or attachment.get("type")
            if att_type and att_type != "FILE":
                print(
                    f"RowId {row_id} (RowNumber {row_number}, VIN {vin}): skipping "
                    f"attachment {attachment.get('id')} of type {att_type}."
                )
                continue

            default_name = attachment.get("name", "attachment")
            file_bytes, filename = download_attachment(
                attachment, fallback_name=default_name
            )
            if file_bytes is None or filename is None:
                continue

            output_path = vin_dir / filename
            with open(output_path, "wb") as f:
                f.write(file_bytes)

            print(f"Saved attachment for VIN {vin}: {output_path}")


if __name__ == "__main__":
    main()
