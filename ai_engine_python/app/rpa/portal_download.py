"""Default to all classes; retain already explicit historical selections."""

import re

from .errors import RpaIssue


def all_scope(options):
    return {
        "mode": "all",
        "class_ids": [],
        "option_names": {option["id"]: option["name"] for option in options},
    }


def selected_scope(target, options, current_class_id=""):
    identified = [option for option in options if not option["all"] and option.get("platform_id")]
    matches = [
        option
        for option in (identified or options)
        if not option["all"]
        and (
            option.get("platform_id") == target.get("class_id")
            if identified
            else option["name"].strip() == target["class_name"].strip()
        )
    ]
    if (
        not matches
        and not identified
        and current_class_id
        and current_class_id == target.get("class_id")
    ):
        matches = [
            option
            for option in options
            if not option["all"] and re.match(r"^(?:导出)?(?:本班级|当前班级)", option["name"])
        ]
    if len(matches) != 1:
        raise RpaIssue("CLASS_CHANGED", "导出班级与选定目录不一致，请更新目录或人工核对")
    return {
        "mode": "selected",
        "class_ids": [matches[0]["id"]],
        "option_names": {option["id"]: option["name"] for option in options},
    }
