"""Read the six retained native crops without network, model, or checkpoint access."""
from pathlib import Path

import numpy as np

from regional_common import checked_bytes, checked_json, decode_batch, require


CASE_IDS = [f"frag{fragment}-context{context}" for fragment in (1, 2, 3) for context in (1, 2)]
PLANES = list(range(7, 58))


class RetainedCrops:
    def __init__(self, raw_pin, cases):
        self.raw = checked_json(raw_pin, 2 * 1024 * 1024)
        require(self.raw["status"] == "PASS" and self.raw["stage"] == "pixels", "Retained acquisition is incomplete")
        require([case["id"] for case in cases] == CASE_IDS, "Frozen six-case order differs")
        self.cases = {case["id"]: case for case in cases}
        retained_cases = {case["case_id"]: case for case in self.raw["fixed_inputs"]["cases"]}
        require(set(retained_cases) == set(self.cases), "Retained crop identities differ")
        for case_id, case in self.cases.items():
            source = retained_cases[case_id]
            require(all(source[key] == case[key] for key in
                        ("bbox_source_xywh", "native_bbox_yxhw", "fragment", "role", "parent_papyrus")),
                    "Retained crop/source contract differs")
        self.contracts = {(item["fragment"], item["plane_index"]): item for item in self.raw["tiff_contracts"]}
        require(len(self.contracts) == len(self.raw["tiff_contracts"]) == 153, "TIFF contract inventory differs")
        require(set(self.contracts) == {(f"frag{f}", z) for f in (1, 2, 3) for z in PLANES}, "TIFF plane identities differ")
        self.schedule = {}
        for index in self.raw["batches"]:
            require(index["fragment"] == index["case_id"].split("-", 1)[0], "Batch fragment/case mismatch")
            key = (index["case_id"], index["plane_index"], index["native_row_start"])
            require(key not in self.schedule, "Duplicate retained batch")
            self.schedule[key] = index
        expected = set()
        for case in cases:
            x, y, width, height = case["bbox_source_xywh"]
            require((width, height) == (1152, 1152), "Unexpected native crop shape")
            for z in PLANES:
                for offset, count in zip((0, 256, 512, 768, 1024), (256, 256, 256, 256, 128)):
                    key = (case["id"], z, y + offset)
                    expected.add(key)
                    require(key in self.schedule and self.schedule[key]["row_count"] == count, "Missing/malformed retained batch")
        require(set(self.schedule) == expected and len(expected) == 1530, "Retained pixel inventory differs")
        self.read_batches = 0
        self.read_body_bytes = 0
        self.read_payload_bytes = 0
        self.input_pins = {raw_pin["path"]: raw_pin}

    def _batch(self, index):
        batch = checked_json(index["batch"], 1024 * 1024)
        self.input_pins[index["batch"]["path"]] = index["batch"]
        contract = self.contracts[(index["fragment"], index["plane_index"])]
        for key in ("request_id", "case_id", "fragment", "plane_index", "native_row_start", "row_count"):
            require(batch[key] == index[key], f"Batch routing differs: {key}")
        require(batch["status_code"] == 206 and batch["source_url"] == contract["url"]
                and batch["etag"] == contract["etag"] and batch["object_total_bytes"] == contract["object_total_bytes"],
                "Batch source identity differs")
        require(contract["orientation"] == 1 and contract["photometric"] == "min-is-black", "TIFF orientation/photometric differs")
        require(batch["content_type"].split(";", 1)[0].strip().lower() == "multipart/byteranges", "Multipart body required")
        body = checked_bytes(batch["multipart"], 1024 * 1024)
        self.input_pins[batch["multipart"]["path"]] = batch["multipart"]
        require(len(batch["parts"]) == len(batch["rows"]) == index["row_count"], "Part/row count differs")
        parts = {tuple(part["range_inclusive"]): part for part in batch["parts"]}
        require(len(parts) == index["row_count"], "Duplicate retained part")
        for row in batch["rows"]:
            span = tuple(row["range_inclusive"])
            require(span in parts, "Row lacks physical part")
            part = parts[span]
            require(all(row[key] == part[key] for key in ("range_inclusive", "payload_offset", "payload_bytes", "payload_sha256")),
                    "Row/part identity differs")
            require(part["headers"].get("content-range") == f"bytes {span[0]}-{span[1]}/{contract['object_total_bytes']}",
                    "Physical part content-range differs")
        self.read_batches += 1
        self.read_body_bytes += len(body)
        self.read_payload_bytes += sum(row["payload_bytes"] for row in batch["rows"])
        return body, batch, contract

    def planes(self, case_id):
        """Yield one 2.53 MiB native plane at a time in original depth order."""
        case = self.cases[case_id]
        y0, _, height, width = case["native_bbox_yxhw"]
        for z in PLANES:
            plane = np.empty((height, width), dtype=np.uint16)
            seen = np.zeros(height, dtype=bool)
            for offset in (0, 256, 512, 768, 1024):
                index = self.schedule[(case_id, z, y0 + offset)]
                body, batch, contract = self._batch(index)
                data = decode_batch(body, batch, contract, case)
                stop = offset + index["row_count"]
                require(not seen[offset:stop].any(), "Crop row assigned twice")
                plane[offset:stop] = data
                seen[offset:stop] = True
            require(seen.all(), "Crop row coverage incomplete")
            yield z, plane

    def summary(self):
        return dict(batch_count=self.read_batches, encoded_body_bytes=self.read_body_bytes,
                    useful_sample_bytes=self.read_payload_bytes,
                    expected_useful_sample_bytes=812187648,
                    original_shape_per_case=[51, 1152, 1152], original_dtype="uint16",
                    original_plane_indices=PLANES,
                    network_calls=0, checkpoint_reads=0, model_calls=0)
