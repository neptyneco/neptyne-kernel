STREAMLIT_PORT = 8501


def base_url_path(shard_id: int, tyne_file_name: str) -> str:
    return f"/ws/{shard_id}/api/sl-app/{tyne_file_name}"


def stream_url_path(shard_id: int, tyne_file_name: str) -> str:
    return f"{base_url_path(shard_id,tyne_file_name)}/_stcore/stream"
