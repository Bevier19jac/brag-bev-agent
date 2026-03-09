from __future__ import annotations

from src.file_processing import (
    PROCESSED_DIR,
    SOURCE_DIR,
    count_supported_source_files,
    ensure_data_directories,
    get_processed_file_count,
    ingest_source_directory,
)


def main() -> int:
    ensure_data_directories()

    print("Brag & Bev local ingestion")
    print(f"Source directory:    {SOURCE_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Supported source files detected: {count_supported_source_files(SOURCE_DIR)}")
    print("")

    summary = ingest_source_directory(SOURCE_DIR, PROCESSED_DIR)

    if summary.messages:
        print("Processing log")
        for message in summary.messages:
            print(f"  {message}")
        print("")

    print("Summary")
    print(f"  Processed: {summary.processed}")
    print(f"  Skipped:   {summary.skipped}")
    print(f"  Errors:    {summary.errors}")
    print(f"  Total processed text files now in data/raw: {get_processed_file_count(PROCESSED_DIR)}")

    if summary.processed_files:
        print("")
        print("Processed outputs")
        for file_name in summary.processed_files:
            print(f"  - {file_name}")

    if summary.skipped_files:
        print("")
        print("Skipped inputs")
        for file_name in summary.skipped_files:
            print(f"  - {file_name}")

    if summary.error_files:
        print("")
        print("Errored inputs")
        for file_name in summary.error_files:
            print(f"  - {file_name}")

    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
