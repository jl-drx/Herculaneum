"""Pure byte-structure parsers. Never interprets image or numerical PPM values."""
import hashlib
import re
import struct
from common import require

TYPE_BYTES = {1:1, 2:1, 3:2, 4:4, 5:8, 6:1, 7:1, 8:2, 9:4, 10:8, 11:4, 12:8}


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def ppm_prefix(data, total):
    require(len(data) == 1024, "PPM prefix must be exactly 1024 bytes")
    marker = re.search(rb"(?:^|\n)<>\n", data)
    require(marker is not None, "PPM header terminator absent within prefix")
    end = marker.end()
    header = data[:end]
    require(b"\r" not in header and b"\0" not in header, "Invalid PPM header framing")
    try:
        text = header.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        require(False, "Invalid UTF-8 PPM header")
    fields, comments = {}, []
    for line in text.split("\n")[:-2]:
        if line.startswith("#"):
            comments.append(line)
            continue
        match = re.fullmatch(r"(width|height|dim|ordered|type|version): (\S+)", line)
        require(match is not None, "Unknown or malformed PPM header field")
        key, value = match.groups()
        require(key not in fields, "Duplicate PPM header field")
        fields[key] = value
    require(set(fields) == {"width", "height", "dim", "ordered", "type", "version"}, "Missing PPM field")
    require(re.fullmatch(r"[1-9][0-9]*", fields["width"]) is not None and
            re.fullmatch(r"[1-9][0-9]*", fields["height"]) is not None, "Invalid PPM dimensions")
    width, height = int(fields["width"]), int(fields["height"])
    require(width >= 11 and height >= 11, "PPM canvas too small")
    require(fields["dim"] == "6" and fields["ordered"] == "true" and
            fields["type"] == "double" and fields["version"] in ("1","1.0"), "Unsupported PPM convention")
    require(total == end + width * height * 48, "PPM object extent disagrees with record layout")
    return dict(width=width, height=height, data_offset=end, record_bytes=48,
                header_text=text, comments=comments, header_sha256=hashlib.sha256(header).hexdigest(),
                fields=fields, numerical_payload_decoded=False,
                uninterpreted_prefix_tail_bytes=1024-end, numerical_byte_order=None,
                source_volume_binding=None, plane_offset_rule=None)


def tiff_header(data, total):
    require(len(data) == 8, "TIFF header must be exactly 8 bytes")
    require(data[:2] in (b"II", b"MM"), "Invalid TIFF byte-order marker")
    order = "<" if data[:2] == b"II" else ">"
    require(struct.unpack_from(order+"H", data, 2)[0] == 42, "Only classic TIFF magic 42 is supported")
    offset = struct.unpack_from(order+"I", data, 4)[0]
    require(8 <= offset and offset+2 <= total, "TIFF IFD pointer outside object")
    return dict(byte_order="little" if order == "<" else "big", struct_order=order, ifd_offset=offset)


def tiff_count(data, header, total):
    require(len(data) == 2, "TIFF IFD count must be exactly 2 bytes")
    count = struct.unpack(header["struct_order"]+"H", data)[0]
    require(1 <= count <= 128, "Unsupported TIFF IFD entry count")
    length = 2+12*count+4
    require(header["ifd_offset"]+length <= total, "TIFF IFD table extends outside object")
    return dict(entries=count, table_bytes=length)


def ascii_description(data):
    require(1 <= len(data) <= 65536 and data.endswith(b"\0"), "TIFF ASCII description needs bounded NUL termination")
    try:
        literal = data.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        require(False, "TIFF description is not ASCII")
    return dict(literal_ascii=literal, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                interpretation="Publisher-supplied literal text only; historical execution and geometry remain unverified.")


def tiff_ifd(data, header, count, total, canvas):
    order, offset = header["struct_order"], header["ifd_offset"]
    require(len(data) == count["table_bytes"], "TIFF IFD table length mismatch")
    require(struct.unpack_from(order+"H", data)[0] == count["entries"], "IFD count changed between requests")
    require(struct.unpack_from(order+"I", data, 2+12*count["entries"])[0] == 0, "TIFF IFD chain/cycle unsupported")
    tags, external, spans = {}, [], [(0,8), (offset,offset+len(data))]
    for i in range(count["entries"]):
        at = 2+12*i
        tag, kind, number = struct.unpack_from(order+"HHI", data, at)
        require(tag not in tags, "Duplicate TIFF tag")
        require(kind in TYPE_BYTES and number > 0, "Unsupported TIFF field type or empty field")
        value = data[at+8:at+12]
        nbytes = TYPE_BYTES[kind]*number
        item = dict(tag=tag, type=kind, count=number, value_bytes_hex=value.hex(), payload_bytes=nbytes)
        if nbytes <= 4:
            item["inline"] = True
        else:
            pointer = struct.unpack(order+"I", value)[0]
            span = (pointer,pointer+nbytes)
            require(8 <= pointer and span[1] <= total, "TIFF field payload outside object")
            require(not any(overlap(span,s) for s in spans), "TIFF field overlaps header/IFD")
            item.update(inline=False, payload_offset=pointer, payload_acquired=False)
            external.append((tag,span))
        tags[tag] = item

    defaults = {}
    def scalar(tag, default=None, allowed_types=(3,4)):
        if tag not in tags:
            require(default is not None, "Required TIFF tag absent: "+str(tag))
            defaults[str(tag)] = default
            return default
        item = tags[tag]
        require(item["type"] in allowed_types and item["count"] == 1 and item["inline"], "Required TIFF tag has an invalid scalar type/count: "+str(tag))
        value = bytes.fromhex(item["value_bytes_hex"])
        return struct.unpack(order+("H" if item["type"] == 3 else "I"),value[:2 if item["type"] == 3 else 4])[0]

    width, height = scalar(256), scalar(257)
    require((width,height) == tuple(canvas), "TIFF dimensions disagree with PPM canvas")
    for tag, expected, default in [(258,16,None),(259,1,1),(262,1,None),(266,1,1),
                                   (274,1,1),(277,1,1),(284,1,1),(317,1,1),(339,1,1)]:
        require(scalar(tag,default,(3,)) == expected, "Unsupported TIFF pixel convention: "+str(tag))
    require(not any(t in tags for t in (322,323,324,325,330,338)), "Tiled, sub-IFD or extra-sample TIFF unsupported")
    strip_offset, strip_bytes, rows = scalar(273), scalar(279), scalar(278,2**32-1)
    require(rows >= height and strip_bytes == width*height*2, "TIFF must contain one complete contiguous uint16 strip")
    strip = (strip_offset,strip_offset+strip_bytes)
    require(8 <= strip_offset and strip[1] <= total and not any(overlap(strip,s) for s in spans), "TIFF strip bounds or structural overlap invalid")
    for tag, span in external:
        require(not overlap(span,strip), "TIFF external field overlaps image strip")
    for i,(tag,span) in enumerate(external):
        require(not any(overlap(span,other) for _,other in external[i+1:]), "TIFF external metadata fields overlap")
    description, description_range = None, None
    if 270 in tags:
        item = tags[270]
        require(item["type"] == 2 and item["count"] <= 65536, "Unsupported TIFF ImageDescription type or size")
        if item["inline"]:
            description = ascii_description(bytes.fromhex(item["value_bytes_hex"])[:item["count"]])
        else:
            description_range = [item["payload_offset"],item["payload_offset"]+item["count"]-1]
    return dict(width=width,height=height,byte_order=header["byte_order"],dtype="uint16",
                ifd_offset=offset,ifd_entries=count["entries"],ifd_bytes=len(data),
                strip_offset=strip_offset,strip_bytes=strip_bytes,rows_per_strip=rows,
                orientation=1,baseline_defaults_applied=defaults,tags=list(tags.values()),
                description=description,description_range=description_range,
                image_pixels_decoded=False, unknown_render_fields=dict(source_volume_uuid=None,
                    ppm_to_native_transform=None,plane_zero_offset=None,plane_offset_rule=None,
                    depth_step_um=None,normal_orientation=None,interpolation=None,renderer_version_and_arguments=None))
