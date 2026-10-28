from PIL import ExifTags


def analyze_metadata(image):
    metadata_result = {
        "has_exif": False,
        "camera_make": "Not found",
        "camera_model": "Not found",
        "lens_model": "Not found",
        "datetime": "Not found",
        "software": "Not found",
        "artist": "Not found",
        "copyright": "Not found",
        "image_description": "Not found",
        "host_computer": "Not found",
        "gps_info": False,
        "orientation": "Not found",
        "x_resolution": "Not found",
        "y_resolution": "Not found",
        "forensic_note": "No metadata available"
    }

    try:
        exif_data = image.getexif()

        if exif_data and len(exif_data) > 0:
            metadata_result["has_exif"] = True

            exif = {}
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif[tag] = value

            metadata_result["camera_make"] = exif.get("Make", "Not found")
            metadata_result["camera_model"] = exif.get("Model", "Not found")
            metadata_result["lens_model"] = exif.get("LensModel", "Not found")
            metadata_result["datetime"] = exif.get(
                "DateTimeOriginal",
                exif.get("DateTimeDigitized", exif.get("DateTime", "Not found"))
            )
            metadata_result["software"] = exif.get("Software", "Not found")
            metadata_result["artist"] = exif.get("Artist", "Not found")
            metadata_result["copyright"] = exif.get("Copyright", "Not found")
            metadata_result["image_description"] = exif.get("ImageDescription", "Not found")
            metadata_result["host_computer"] = exif.get("HostComputer", "Not found")
            metadata_result["orientation"] = exif.get("Orientation", "Not found")
            metadata_result["x_resolution"] = exif.get("XResolution", "Not found")
            metadata_result["y_resolution"] = exif.get("YResolution", "Not found")
            metadata_result["gps_info"] = "GPSInfo" in exif

            # Smarter forensic interpretation
            camera_fields_found = (
                metadata_result["camera_make"] != "Not found" or
                metadata_result["camera_model"] != "Not found" or
                metadata_result["lens_model"] != "Not found"
            )

            software_found = metadata_result["software"] != "Not found"

            if camera_fields_found and not software_found:
                metadata_result["forensic_note"] = "Natural camera EXIF metadata detected"
            elif camera_fields_found and software_found:
                metadata_result["forensic_note"] = "Camera metadata found, but software/editing metadata is also present"
            elif not camera_fields_found and software_found:
                metadata_result["forensic_note"] = "No camera EXIF found; software/export metadata detected"
            elif metadata_result["has_exif"]:
                metadata_result["forensic_note"] = "Metadata exists but important camera fields are missing"

        else:
            metadata_result["forensic_note"] = "No EXIF metadata found"

    except Exception as e:
        metadata_result["forensic_note"] = f"Metadata analysis failed: {str(e)}"

    return metadata_result