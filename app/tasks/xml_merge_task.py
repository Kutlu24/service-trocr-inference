from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter
from app.models.raw_xml_upload_models import RawXMLMergeResponseModel

async def xml_merge_task(
        huggingface_token: str,
        created_status: RawXMLMergeResponseModel
):
    print("XML merge started")

    worker = InferenceToRawXMLWriter(
        raw_xml_repo=created_status.source_raw_repo,
        inference_repo=created_status.source_inference_repo,
        token=huggingface_token,
    )

    try:
        worker.process_and_upload(
            output_repo=created_status.output_repo_name
        )
        state = "completed"
    except Exception as e:
        print(f"XML merge failed: {e}")
        state = "failed"

    # write to status file...
