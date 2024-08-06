class _Unspecified:
    def __repr__(self):
        return "None"


_UNSPECIFIED = _Unspecified()


def _request(method, body, mappable_types):
    from ...dash import Dash

    body = {k: v for k, v in body.items() if v is not _UNSPECIFIED}
    return Dash.instance().google_sheets_request(method, body, mappable_types)


def update_spreadsheet_properties(properties=_UNSPECIFIED, fields="*"):
    """Updates properties of a spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateSpreadsheetPropertiesRequest)
    """
    body = {"properties": properties, "fields": fields}

    mappable_types = {}

    return _request("updateSpreadsheetProperties", body, mappable_types)


def update_sheet_properties(properties=_UNSPECIFIED, fields="*"):
    """Updates properties of the sheet with the specified sheetId.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateSheetPropertiesRequest)
    """
    body = {"properties": properties, "fields": fields}

    mappable_types = {}

    return _request("updateSheetProperties", body, mappable_types)


def update_dimension_properties(
    range=_UNSPECIFIED,
    data_source_sheet_range=_UNSPECIFIED,
    properties=_UNSPECIFIED,
    fields="*",
):
    """Updates properties of dimensions within the specified range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateDimensionPropertiesRequest)
    """
    body = {
        "range": range,
        "dataSourceSheetRange": data_source_sheet_range,
        "properties": properties,
        "fields": fields,
    }

    mappable_types = {}

    return _request("updateDimensionProperties", body, mappable_types)


def update_named_range(named_range=_UNSPECIFIED, fields="*"):
    """Updates properties of the named range with the specified namedRangeId.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateNamedRangeRequest)
    """
    body = {"namedRange": named_range, "fields": fields}

    mappable_types = {"namedRange.range": "GridRange"}

    return _request("updateNamedRange", body, mappable_types)


def repeat_cell(range=_UNSPECIFIED, cell=_UNSPECIFIED, fields="*"):
    """Updates all cells in the range to the values in the given Cell object. Only the fields listed in the fields field are updated; others are unchanged. If writing a cell with a formula, the formula's ranges will automatically increment for each field in the range. For example, if writing a cell with formula `=A1` into range B2:C4, B2 would be `=A1`, B3 would be `=A2`, B4 would be `=A3`, C2 would be `=B1`, C3 would be `=B2`, C4 would be `=B3`. To keep the formula's ranges static, use the `$` indicator. For example, use the formula `=$A$1` to prevent both the row and the column from incrementing.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#RepeatCellRequest)
    """
    body = {"range": range, "cell": cell, "fields": fields}

    mappable_types = {"range": "GridRange", "cell.pivotTable.source": "GridRange"}

    return _request("repeatCell", body, mappable_types)


def add_named_range(named_range=_UNSPECIFIED):
    """Adds a named range to the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddNamedRangeRequest)
    """
    body = {"namedRange": named_range}

    mappable_types = {"namedRange.range": "GridRange"}

    return _request("addNamedRange", body, mappable_types)


def delete_named_range(named_range_id=_UNSPECIFIED):
    """Removes the named range with the given ID from the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteNamedRangeRequest)
    """
    body = {"namedRangeId": named_range_id}

    mappable_types = {}

    return _request("deleteNamedRange", body, mappable_types)


def add_sheet(properties=_UNSPECIFIED):
    """Adds a new sheet. When a sheet is added at a given index, all subsequent sheets' indexes are incremented. To add an object sheet, use AddChartRequest instead and specify EmbeddedObjectPosition.sheetId or EmbeddedObjectPosition.newSheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddSheetRequest)
    """
    body = {"properties": properties}

    mappable_types = {}

    return _request("addSheet", body, mappable_types)


def delete_sheet(sheet_id=_UNSPECIFIED):
    """Deletes the requested sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteSheetRequest)
    """
    body = {"sheetId": sheet_id}

    mappable_types = {}

    return _request("deleteSheet", body, mappable_types)


def auto_fill(
    range=_UNSPECIFIED,
    source_and_destination=_UNSPECIFIED,
    use_alternate_series=_UNSPECIFIED,
):
    """Fills in more data based on existing data.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AutoFillRequest)
    """
    body = {
        "range": range,
        "sourceAndDestination": source_and_destination,
        "useAlternateSeries": use_alternate_series,
    }

    mappable_types = {"range": "GridRange", "sourceAndDestination.source": "GridRange"}

    return _request("autoFill", body, mappable_types)


def cut_paste(source=_UNSPECIFIED, destination=_UNSPECIFIED, paste_type=_UNSPECIFIED):
    """Moves data from the source to the destination.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#CutPasteRequest)
    """
    body = {"source": source, "destination": destination, "pasteType": paste_type}

    mappable_types = {"source": "GridRange", "destination": "GridCoordinate"}

    return _request("cutPaste", body, mappable_types)


def copy_paste(
    source=_UNSPECIFIED,
    destination=_UNSPECIFIED,
    paste_type=_UNSPECIFIED,
    paste_orientation=_UNSPECIFIED,
):
    """Copies data from the source to the destination.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#CopyPasteRequest)
    """
    body = {
        "source": source,
        "destination": destination,
        "pasteType": paste_type,
        "pasteOrientation": paste_orientation,
    }

    mappable_types = {"source": "GridRange", "destination": "GridRange"}

    return _request("copyPaste", body, mappable_types)


def merge_cells(range=_UNSPECIFIED, merge_type=_UNSPECIFIED):
    """Merges all cells in the range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#MergeCellsRequest)
    """
    body = {"range": range, "mergeType": merge_type}

    mappable_types = {"range": "GridRange"}

    return _request("mergeCells", body, mappable_types)


def unmerge_cells(range=_UNSPECIFIED):
    """Unmerges cells in the given range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UnmergeCellsRequest)
    """
    body = {"range": range}

    mappable_types = {"range": "GridRange"}

    return _request("unmergeCells", body, mappable_types)


def update_borders(
    range=_UNSPECIFIED,
    top=_UNSPECIFIED,
    bottom=_UNSPECIFIED,
    left=_UNSPECIFIED,
    right=_UNSPECIFIED,
    inner_horizontal=_UNSPECIFIED,
    inner_vertical=_UNSPECIFIED,
):
    """Updates the borders of a range. If a field is not set in the request, that means the border remains as-is. For example, with two subsequent UpdateBordersRequest: 1. range: A1:A5 `{ top: RED, bottom: WHITE }` 2. range: A1:A5 `{ left: BLUE }` That would result in A1:A5 having a borders of `{ top: RED, bottom: WHITE, left: BLUE }`. If you want to clear a border, explicitly set the style to NONE.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateBordersRequest)
    """
    body = {
        "range": range,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "innerHorizontal": inner_horizontal,
        "innerVertical": inner_vertical,
    }

    mappable_types = {"range": "GridRange"}

    return _request("updateBorders", body, mappable_types)


def update_cells(start=_UNSPECIFIED, range=_UNSPECIFIED, rows=_UNSPECIFIED, fields="*"):
    """Updates all cells in a range with new data.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateCellsRequest)
    """
    body = {"start": start, "range": range, "rows": rows, "fields": fields}

    mappable_types = {"start": "GridCoordinate", "range": "GridRange"}

    return _request("updateCells", body, mappable_types)


def add_filter_view(filter=_UNSPECIFIED):
    """Adds a filter view.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddFilterViewRequest)
    """
    body = {"filter": filter}

    mappable_types = {"filter.range": "GridRange"}

    return _request("addFilterView", body, mappable_types)


def append_cells(sheet_id=_UNSPECIFIED, rows=_UNSPECIFIED, fields="*"):
    """Adds new cells after the last row with data in a sheet, inserting new rows into the sheet if necessary.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AppendCellsRequest)
    """
    body = {"sheetId": sheet_id, "rows": rows, "fields": fields}

    mappable_types = {}

    return _request("appendCells", body, mappable_types)


def clear_basic_filter(sheet_id=_UNSPECIFIED):
    """Clears the basic filter, if any exists on the sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#ClearBasicFilterRequest)
    """
    body = {"sheetId": sheet_id}

    mappable_types = {}

    return _request("clearBasicFilter", body, mappable_types)


def delete_dimension(range=_UNSPECIFIED):
    """Deletes the dimensions from the sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteDimensionRequest)
    """
    body = {"range": range}

    mappable_types = {}

    return _request("deleteDimension", body, mappable_types)


def delete_embedded_object(object_id=_UNSPECIFIED):
    """Deletes the embedded object with the given ID.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteEmbeddedObjectRequest)
    """
    body = {"objectId": object_id}

    mappable_types = {}

    return _request("deleteEmbeddedObject", body, mappable_types)


def delete_filter_view(filter_id=_UNSPECIFIED):
    """Deletes a particular filter view.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteFilterViewRequest)
    """
    body = {"filterId": filter_id}

    mappable_types = {}

    return _request("deleteFilterView", body, mappable_types)


def duplicate_filter_view(filter_id=_UNSPECIFIED):
    """Duplicates a particular filter view.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DuplicateFilterViewRequest)
    """
    body = {"filterId": filter_id}

    mappable_types = {}

    return _request("duplicateFilterView", body, mappable_types)


def duplicate_sheet(
    source_sheet_id=_UNSPECIFIED,
    insert_sheet_index=_UNSPECIFIED,
    new_sheet_id=_UNSPECIFIED,
    new_sheet_name=_UNSPECIFIED,
):
    """Duplicates the contents of a sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DuplicateSheetRequest)
    """
    body = {
        "sourceSheetId": source_sheet_id,
        "insertSheetIndex": insert_sheet_index,
        "newSheetId": new_sheet_id,
        "newSheetName": new_sheet_name,
    }

    mappable_types = {}

    return _request("duplicateSheet", body, mappable_types)


def find_replace(
    find=_UNSPECIFIED,
    replacement=_UNSPECIFIED,
    range=_UNSPECIFIED,
    sheet_id=_UNSPECIFIED,
    all_sheets=_UNSPECIFIED,
    match_case=_UNSPECIFIED,
    match_entire_cell=_UNSPECIFIED,
    search_by_regex=_UNSPECIFIED,
    include_formulas=_UNSPECIFIED,
):
    """Finds and replaces data in cells over a range, sheet, or all sheets.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#FindReplaceRequest)
    """
    body = {
        "find": find,
        "replacement": replacement,
        "range": range,
        "sheetId": sheet_id,
        "allSheets": all_sheets,
        "matchCase": match_case,
        "matchEntireCell": match_entire_cell,
        "searchByRegex": search_by_regex,
        "includeFormulas": include_formulas,
    }

    mappable_types = {"range": "GridRange"}

    return _request("findReplace", body, mappable_types)


def insert_dimension(range=_UNSPECIFIED, inherit_from_before=_UNSPECIFIED):
    """Inserts rows or columns in a sheet at a particular index.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#InsertDimensionRequest)
    """
    body = {"range": range, "inheritFromBefore": inherit_from_before}

    mappable_types = {}

    return _request("insertDimension", body, mappable_types)


def insert_range(range=_UNSPECIFIED, shift_dimension=_UNSPECIFIED):
    """Inserts cells into a range, shifting the existing cells over or down.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#InsertRangeRequest)
    """
    body = {"range": range, "shiftDimension": shift_dimension}

    mappable_types = {"range": "GridRange"}

    return _request("insertRange", body, mappable_types)


def move_dimension(source=_UNSPECIFIED, destination_index=_UNSPECIFIED):
    """Moves one or more rows or columns.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#MoveDimensionRequest)
    """
    body = {"source": source, "destinationIndex": destination_index}

    mappable_types = {}

    return _request("moveDimension", body, mappable_types)


def update_embedded_object_position(
    object_id=_UNSPECIFIED, new_position=_UNSPECIFIED, fields="*"
):
    """Update an embedded object's position (such as a moving or resizing a chart or image).

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateEmbeddedObjectPositionRequest)
    """
    body = {"objectId": object_id, "newPosition": new_position, "fields": fields}

    mappable_types = {"newPosition.overlayPosition.anchorCell": "GridCoordinate"}

    return _request("updateEmbeddedObjectPosition", body, mappable_types)


def paste_data(
    coordinate=_UNSPECIFIED,
    data=_UNSPECIFIED,
    delimiter=_UNSPECIFIED,
    html=_UNSPECIFIED,
    type=_UNSPECIFIED,
):
    """Inserts data into the spreadsheet starting at the specified coordinate.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#PasteDataRequest)
    """
    body = {
        "coordinate": coordinate,
        "data": data,
        "delimiter": delimiter,
        "html": html,
        "type": type,
    }

    mappable_types = {"coordinate": "GridCoordinate"}

    return _request("pasteData", body, mappable_types)


def text_to_columns(
    source=_UNSPECIFIED, delimiter=_UNSPECIFIED, delimiter_type=_UNSPECIFIED
):
    """Splits a column of text into multiple columns, based on a delimiter in each cell.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#TextToColumnsRequest)
    """
    body = {"source": source, "delimiter": delimiter, "delimiterType": delimiter_type}

    mappable_types = {"source": "GridRange"}

    return _request("textToColumns", body, mappable_types)


def update_filter_view(filter=_UNSPECIFIED, fields="*"):
    """Updates properties of the filter view.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateFilterViewRequest)
    """
    body = {"filter": filter, "fields": fields}

    mappable_types = {"filter.range": "GridRange"}

    return _request("updateFilterView", body, mappable_types)


def delete_range(range=_UNSPECIFIED, shift_dimension=_UNSPECIFIED):
    """Deletes a range of cells, shifting other cells into the deleted area.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteRangeRequest)
    """
    body = {"range": range, "shiftDimension": shift_dimension}

    mappable_types = {"range": "GridRange"}

    return _request("deleteRange", body, mappable_types)


def append_dimension(
    sheet_id=_UNSPECIFIED, dimension=_UNSPECIFIED, length=_UNSPECIFIED
):
    """Appends rows or columns to the end of a sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AppendDimensionRequest)
    """
    body = {"sheetId": sheet_id, "dimension": dimension, "length": length}

    mappable_types = {}

    return _request("appendDimension", body, mappable_types)


def add_conditional_format_rule(rule=_UNSPECIFIED, index=_UNSPECIFIED):
    """Adds a new conditional format rule at the given index. All subsequent rules' indexes are incremented.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddConditionalFormatRuleRequest)
    """
    body = {"rule": rule, "index": index}

    mappable_types = {}

    return _request("addConditionalFormatRule", body, mappable_types)


def update_conditional_format_rule(
    rule=_UNSPECIFIED, new_index=_UNSPECIFIED, index=_UNSPECIFIED, sheet_id=_UNSPECIFIED
):
    """Updates a conditional format rule at the given index, or moves a conditional format rule to another index.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateConditionalFormatRuleRequest)
    """
    body = {"rule": rule, "newIndex": new_index, "index": index, "sheetId": sheet_id}

    mappable_types = {}

    return _request("updateConditionalFormatRule", body, mappable_types)


def delete_conditional_format_rule(index=_UNSPECIFIED, sheet_id=_UNSPECIFIED):
    """Deletes a conditional format rule at the given index. All subsequent rules' indexes are decremented.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteConditionalFormatRuleRequest)
    """
    body = {"index": index, "sheetId": sheet_id}

    mappable_types = {}

    return _request("deleteConditionalFormatRule", body, mappable_types)


def sort_range(range=_UNSPECIFIED, sort_specs=_UNSPECIFIED):
    """Sorts data in rows based on a sort order per column.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#SortRangeRequest)
    """
    body = {"range": range, "sortSpecs": sort_specs}

    mappable_types = {"range": "GridRange"}

    return _request("sortRange", body, mappable_types)


def set_data_validation(range=_UNSPECIFIED, rule=_UNSPECIFIED):
    """Sets a data validation rule to every cell in the range. To clear validation in a range, call this with no rule specified.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#SetDataValidationRequest)
    """
    body = {"range": range, "rule": rule}

    mappable_types = {"range": "GridRange"}

    return _request("setDataValidation", body, mappable_types)


def set_basic_filter(filter=_UNSPECIFIED):
    """Sets the basic filter associated with a sheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#SetBasicFilterRequest)
    """
    body = {"filter": filter}

    mappable_types = {"filter.range": "GridRange"}

    return _request("setBasicFilter", body, mappable_types)


def add_protected_range(protected_range=_UNSPECIFIED):
    """Adds a new protected range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddProtectedRangeRequest)
    """
    body = {"protectedRange": protected_range}

    mappable_types = {"protectedRange.range": "GridRange"}

    return _request("addProtectedRange", body, mappable_types)


def update_protected_range(protected_range=_UNSPECIFIED, fields="*"):
    """Updates an existing protected range with the specified protectedRangeId.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateProtectedRangeRequest)
    """
    body = {"protectedRange": protected_range, "fields": fields}

    mappable_types = {"protectedRange.range": "GridRange"}

    return _request("updateProtectedRange", body, mappable_types)


def delete_protected_range(protected_range_id=_UNSPECIFIED):
    """Deletes the protected range with the given ID.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteProtectedRangeRequest)
    """
    body = {"protectedRangeId": protected_range_id}

    mappable_types = {}

    return _request("deleteProtectedRange", body, mappable_types)


def auto_resize_dimensions(
    dimensions=_UNSPECIFIED, data_source_sheet_dimensions=_UNSPECIFIED
):
    """Automatically resizes one or more dimensions based on the contents of the cells in that dimension.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AutoResizeDimensionsRequest)
    """
    body = {
        "dimensions": dimensions,
        "dataSourceSheetDimensions": data_source_sheet_dimensions,
    }

    mappable_types = {}

    return _request("autoResizeDimensions", body, mappable_types)


def add_chart(chart=_UNSPECIFIED):
    """Adds a chart to a sheet in the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddChartRequest)
    """
    body = {"chart": chart}

    mappable_types = {"chart.position.overlayPosition.anchorCell": "GridCoordinate"}

    return _request("addChart", body, mappable_types)


def update_chart_spec(chart_id=_UNSPECIFIED, spec=_UNSPECIFIED):
    """Updates a chart's specifications. (This does not move or resize a chart. To move or resize a chart, use UpdateEmbeddedObjectPositionRequest.)

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateChartSpecRequest)
    """
    body = {"chartId": chart_id, "spec": spec}

    mappable_types = {}

    return _request("updateChartSpec", body, mappable_types)


def update_banding(banded_range=_UNSPECIFIED, fields="*"):
    """Updates properties of the supplied banded range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateBandingRequest)
    """
    body = {"bandedRange": banded_range, "fields": fields}

    mappable_types = {"bandedRange.range": "GridRange"}

    return _request("updateBanding", body, mappable_types)


def add_banding(banded_range=_UNSPECIFIED):
    """Adds a new banded range to the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddBandingRequest)
    """
    body = {"bandedRange": banded_range}

    mappable_types = {"bandedRange.range": "GridRange"}

    return _request("addBanding", body, mappable_types)


def delete_banding(banded_range_id=_UNSPECIFIED):
    """Removes the banded range with the given ID from the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteBandingRequest)
    """
    body = {"bandedRangeId": banded_range_id}

    mappable_types = {}

    return _request("deleteBanding", body, mappable_types)


def create_developer_metadata(developer_metadata=_UNSPECIFIED):
    """A request to create developer metadata.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#CreateDeveloperMetadataRequest)
    """
    body = {"developerMetadata": developer_metadata}

    mappable_types = {}

    return _request("createDeveloperMetadata", body, mappable_types)


def update_developer_metadata(
    data_filters=_UNSPECIFIED, developer_metadata=_UNSPECIFIED, fields="*"
):
    """A request to update properties of developer metadata. Updates the properties of the developer metadata selected by the filters to the values provided in the DeveloperMetadata resource. Callers must specify the properties they wish to update in the fields parameter, as well as specify at least one DataFilter matching the metadata they wish to update.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateDeveloperMetadataRequest)
    """
    body = {
        "dataFilters": data_filters,
        "developerMetadata": developer_metadata,
        "fields": fields,
    }

    mappable_types = {}

    return _request("updateDeveloperMetadata", body, mappable_types)


def delete_developer_metadata(data_filter=_UNSPECIFIED):
    """A request to delete developer metadata.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteDeveloperMetadataRequest)
    """
    body = {"dataFilter": data_filter}

    mappable_types = {"dataFilter.gridRange": "GridRange"}

    return _request("deleteDeveloperMetadata", body, mappable_types)


def randomize_range(range=_UNSPECIFIED):
    """Randomizes the order of the rows in a range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#RandomizeRangeRequest)
    """
    body = {"range": range}

    mappable_types = {"range": "GridRange"}

    return _request("randomizeRange", body, mappable_types)


def add_dimension_group(range=_UNSPECIFIED):
    """Creates a group over the specified range. If the requested range is a superset of the range of an existing group G, then the depth of G is incremented and this new group G' has the depth of that group. For example, a group [C:D, depth 1] + [B:E] results in groups [B:E, depth 1] and [C:D, depth 2]. If the requested range is a subset of the range of an existing group G, then the depth of the new group G' becomes one greater than the depth of G. For example, a group [B:E, depth 1] + [C:D] results in groups [B:E, depth 1] and [C:D, depth 2]. If the requested range starts before and ends within, or starts within and ends after, the range of an existing group G, then the range of the existing group G becomes the union of the ranges, and the new group G' has depth one greater than the depth of G and range as the intersection of the ranges. For example, a group [B:D, depth 1] + [C:E] results in groups [B:E, depth 1] and [C:D, depth 2].

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddDimensionGroupRequest)
    """
    body = {"range": range}

    mappable_types = {}

    return _request("addDimensionGroup", body, mappable_types)


def delete_dimension_group(range=_UNSPECIFIED):
    """Deletes a group over the specified range by decrementing the depth of the dimensions in the range. For example, assume the sheet has a depth-1 group over B:E and a depth-2 group over C:D. Deleting a group over D:E leaves the sheet with a depth-1 group over B:D and a depth-2 group over C:C.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteDimensionGroupRequest)
    """
    body = {"range": range}

    mappable_types = {}

    return _request("deleteDimensionGroup", body, mappable_types)


def update_dimension_group(dimension_group=_UNSPECIFIED, fields="*"):
    """Updates the state of the specified group.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateDimensionGroupRequest)
    """
    body = {"dimensionGroup": dimension_group, "fields": fields}

    mappable_types = {}

    return _request("updateDimensionGroup", body, mappable_types)


def trim_whitespace(range=_UNSPECIFIED):
    """Trims the whitespace (such as spaces, tabs, or new lines) in every cell in the specified range. This request removes all whitespace from the start and end of each cell's text, and reduces any subsequence of remaining whitespace characters to a single space. If the resulting trimmed text starts with a '+' or '=' character, the text remains as a string value and isn't interpreted as a formula.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#TrimWhitespaceRequest)
    """
    body = {"range": range}

    mappable_types = {"range": "GridRange"}

    return _request("trimWhitespace", body, mappable_types)


def delete_duplicates(range=_UNSPECIFIED, comparison_columns=_UNSPECIFIED):
    """Removes rows within this range that contain values in the specified columns that are duplicates of values in any previous row. Rows with identical values but different letter cases, formatting, or formulas are considered to be duplicates. This request also removes duplicate rows hidden from view (for example, due to a filter). When removing duplicates, the first instance of each duplicate row scanning from the top downwards is kept in the resulting range. Content outside of the specified range isn't removed, and rows considered duplicates do not have to be adjacent to each other in the range.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteDuplicatesRequest)
    """
    body = {"range": range, "comparisonColumns": comparison_columns}

    mappable_types = {"range": "GridRange"}

    return _request("deleteDuplicates", body, mappable_types)


def update_embedded_object_border(
    object_id=_UNSPECIFIED, border=_UNSPECIFIED, fields="*"
):
    """Updates an embedded object's border property.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateEmbeddedObjectBorderRequest)
    """
    body = {"objectId": object_id, "border": border, "fields": fields}

    mappable_types = {}

    return _request("updateEmbeddedObjectBorder", body, mappable_types)


def add_slicer(slicer=_UNSPECIFIED):
    """Adds a slicer to a sheet in the spreadsheet.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddSlicerRequest)
    """
    body = {"slicer": slicer}

    mappable_types = {
        "slicer.spec.dataRange": "GridRange",
        "slicer.position.overlayPosition.anchorCell": "GridCoordinate",
    }

    return _request("addSlicer", body, mappable_types)


def update_slicer_spec(slicer_id=_UNSPECIFIED, spec=_UNSPECIFIED, fields="*"):
    """Updates a slicer's specifications. (This does not move or resize a slicer. To move or resize a slicer use UpdateEmbeddedObjectPositionRequest.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateSlicerSpecRequest)
    """
    body = {"slicerId": slicer_id, "spec": spec, "fields": fields}

    mappable_types = {"spec.dataRange": "GridRange"}

    return _request("updateSlicerSpec", body, mappable_types)


def add_data_source(data_source=_UNSPECIFIED):
    """Adds a data source. After the data source is added successfully, an associated DATA_SOURCE sheet is created and an execution is triggered to refresh the sheet to read data from the data source. The request requires an additional `bigquery.readonly` OAuth scope.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#AddDataSourceRequest)
    """
    body = {"dataSource": data_source}

    mappable_types = {}

    return _request("addDataSource", body, mappable_types)


def update_data_source(data_source=_UNSPECIFIED, fields="*"):
    """Updates a data source. After the data source is updated successfully, an execution is triggered to refresh the associated DATA_SOURCE sheet to read data from the updated data source. The request requires an additional `bigquery.readonly` OAuth scope.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#UpdateDataSourceRequest)
    """
    body = {"dataSource": data_source, "fields": fields}

    mappable_types = {}

    return _request("updateDataSource", body, mappable_types)


def delete_data_source(data_source_id=_UNSPECIFIED):
    """Deletes a data source. The request also deletes the associated data source sheet, and unlinks all associated data source objects.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#DeleteDataSourceRequest)
    """
    body = {"dataSourceId": data_source_id}

    mappable_types = {}

    return _request("deleteDataSource", body, mappable_types)


def refresh_data_source(
    references=_UNSPECIFIED,
    data_source_id=_UNSPECIFIED,
    is_all=_UNSPECIFIED,
    force=_UNSPECIFIED,
):
    """Refreshes one or multiple data source objects in the spreadsheet by the specified references. The request requires an additional `bigquery.readonly` OAuth scope. If there are multiple refresh requests referencing the same data source objects in one batch, only the last refresh request is processed, and all those requests will have the same response accordingly.

    [Link](https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/request#RefreshDataSourceRequest)
    """
    body = {
        "references": references,
        "dataSourceId": data_source_id,
        "isAll": is_all,
        "force": force,
    }

    mappable_types = {}

    return _request("refreshDataSource", body, mappable_types)


__all__ = [
    "update_spreadsheet_properties",
    "update_sheet_properties",
    "update_dimension_properties",
    "update_named_range",
    "repeat_cell",
    "add_named_range",
    "delete_named_range",
    "add_sheet",
    "delete_sheet",
    "auto_fill",
    "cut_paste",
    "copy_paste",
    "merge_cells",
    "unmerge_cells",
    "update_borders",
    "update_cells",
    "add_filter_view",
    "append_cells",
    "clear_basic_filter",
    "delete_dimension",
    "delete_embedded_object",
    "delete_filter_view",
    "duplicate_filter_view",
    "duplicate_sheet",
    "find_replace",
    "insert_dimension",
    "insert_range",
    "move_dimension",
    "update_embedded_object_position",
    "paste_data",
    "text_to_columns",
    "update_filter_view",
    "delete_range",
    "append_dimension",
    "add_conditional_format_rule",
    "update_conditional_format_rule",
    "delete_conditional_format_rule",
    "sort_range",
    "set_data_validation",
    "set_basic_filter",
    "add_protected_range",
    "update_protected_range",
    "delete_protected_range",
    "auto_resize_dimensions",
    "add_chart",
    "update_chart_spec",
    "update_banding",
    "add_banding",
    "delete_banding",
    "create_developer_metadata",
    "update_developer_metadata",
    "delete_developer_metadata",
    "randomize_range",
    "add_dimension_group",
    "delete_dimension_group",
    "update_dimension_group",
    "trim_whitespace",
    "delete_duplicates",
    "update_embedded_object_border",
    "add_slicer",
    "update_slicer_spec",
    "add_data_source",
    "update_data_source",
    "delete_data_source",
    "refresh_data_source",
]
