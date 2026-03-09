from __future__ import annotations

from src.file_processing import PROCESSED_DIR, SOURCE_DIR, ensure_data_directories, ingest_source_directory


def main() -> int:
    ensure_data_directories()
    summary = ingest_source_directory(SOURCE_DIR, PROCESSED_DIR)

    print("Brag & Bev local ingestion")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print("")

    if summary.messages:
        for message in summary.messages:
            print(message)
        print("")

    print("Summary")
    print(f"  Processed: {summary.processed}")
    print(f"  Skipped:   {summary.skipped}")
    print(f"  Errors:    {summary.errors}")

    if summary.processed_files:
        print("")
        print("Processed files")
        for file_name in summary.processed_files:
            print(f"  - {file_name}")

    if summary.skipped_files:
        print("")
        print("Skipped files")
        for file_name in summary.skipped_files:
            print(f"  - {file_name}")

    if summary.error_files:
        print("")
        print("Errored files")
        for file_name in summary.error_files:
            print(f"  - {file_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
