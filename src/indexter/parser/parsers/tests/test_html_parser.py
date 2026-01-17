from unittest.mock import Mock

import pytest
from tree_sitter import Node

from indexter.parser.parsers.html import HtmlParser
from indexter.walker.models import Document, DocumentMetadata


@pytest.fixture
def html_parser():
    """Create an HtmlParser instance for testing."""
    return HtmlParser()


@pytest.fixture
def sample_html_document():
    """Create a sample HTML Document for testing."""
    content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is a test.</p>
</body>
</html>
"""
    metadata = DocumentMetadata(
        repo="test-repo",
        repo_path="/path/to/repo",
        hash="abc123",
        ext=".html",
        size_bytes=len(content),
        mtime=1234567890.0,
    )
    return Document(
        path="test.html",
        content=content,
        metadata=metadata,
    )


# Unit tests for helper methods


class TestExtractTextContent:
    """Test the _extract_text_content helper method."""

    def test_should_extract_text_from_simple_element(self, html_parser):
        text_node = Mock(spec=Node)
        text_node.type = "text"
        text_node.text = b"Hello World"
        text_node.children = []

        node = Mock(spec=Node)
        node.type = "element"
        node.children = [text_node]

        result = html_parser._extract_text_content(node, b"")
        assert result == "Hello World"

    def test_should_extract_text_from_nested_elements(self, html_parser):
        inner_text = Mock(spec=Node)
        inner_text.type = "text"
        inner_text.text = b"inner"
        inner_text.children = []

        inner_element = Mock(spec=Node)
        inner_element.type = "element"
        inner_element.children = [inner_text]

        outer_text = Mock(spec=Node)
        outer_text.type = "text"
        outer_text.text = b"outer"
        outer_text.children = []

        node = Mock(spec=Node)
        node.type = "element"
        node.children = [outer_text, inner_element]

        result = html_parser._extract_text_content(node, b"")
        assert "outer" in result
        assert "inner" in result

    def test_should_skip_non_text_nodes(self, html_parser):
        tag_node = Mock(spec=Node)
        tag_node.type = "start_tag"
        tag_node.children = []

        text_node = Mock(spec=Node)
        text_node.type = "text"
        text_node.text = b"content"
        text_node.children = []

        node = Mock(spec=Node)
        node.type = "element"
        node.children = [tag_node, text_node]

        result = html_parser._extract_text_content(node, b"")
        assert result == "content"

    def test_should_handle_node_with_no_text(self, html_parser):
        tag_node = Mock(spec=Node)
        tag_node.type = "start_tag"
        tag_node.children = []

        node = Mock(spec=Node)
        node.type = "element"
        node.children = [tag_node]

        result = html_parser._extract_text_content(node, b"")
        assert result == ""

    def test_should_join_multiple_text_nodes(self, html_parser):
        text1 = Mock(spec=Node)
        text1.type = "text"
        text1.text = b"First"
        text1.children = []

        text2 = Mock(spec=Node)
        text2.type = "text"
        text2.text = b"Second"
        text2.children = []

        node = Mock(spec=Node)
        node.type = "element"
        node.children = [text1, text2]

        result = html_parser._extract_text_content(node, b"")
        assert result == "First Second"


class TestNormalizeText:
    """Test the _normalize_text helper method."""

    def test_should_convert_to_lowercase(self, html_parser):
        result = html_parser._normalize_text("Hello WORLD")
        assert result == "hello world"

    def test_should_remove_multiple_spaces(self, html_parser):
        result = html_parser._normalize_text("hello    world")
        assert result == "hello world"

    def test_should_remove_newlines(self, html_parser):
        result = html_parser._normalize_text("hello\nworld\r\ntest")
        assert result == "hello world test"

    def test_should_remove_tabs(self, html_parser):
        result = html_parser._normalize_text("hello\t\tworld")
        assert result == "hello world"

    def test_should_strip_leading_and_trailing_whitespace(self, html_parser):
        result = html_parser._normalize_text("  hello world  ")
        assert result == "hello world"

    def test_should_handle_empty_string(self, html_parser):
        result = html_parser._normalize_text("")
        assert result == ""

    def test_should_handle_whitespace_only_string(self, html_parser):
        result = html_parser._normalize_text("   \n\t  ")
        assert result == ""


class TestRemoveStopwords:
    """Test the _remove_stopwords helper method."""

    def test_should_remove_common_stopwords(self, html_parser):
        result = html_parser._remove_stopwords("the cat is on the mat")
        assert result == "cat mat"

    def test_should_keep_non_stopwords(self, html_parser):
        result = html_parser._remove_stopwords("python programming language")
        assert result == "python programming language"

    def test_should_handle_mixed_case_after_normalization(self, html_parser):
        # Note: This method expects normalized (lowercase) input
        result = html_parser._remove_stopwords("hello world and goodbye")
        assert result == "hello world goodbye"

    def test_should_handle_empty_string(self, html_parser):
        result = html_parser._remove_stopwords("")
        assert result == ""

    def test_should_handle_only_stopwords(self, html_parser):
        result = html_parser._remove_stopwords("the and is a")
        assert result == ""

    def test_should_preserve_word_order(self, html_parser):
        result = html_parser._remove_stopwords("apple banana cherry")
        assert result == "apple banana cherry"


class TestGenerateNodeName:
    """Test the _generate_node_name helper method."""

    def test_should_use_first_five_words_for_headers(self, html_parser):
        result = html_parser._generate_node_name("one two three four five six seven", "h1")
        assert result == "one two three four five"

    def test_should_use_all_words_if_less_than_five_for_headers(self, html_parser):
        result = html_parser._generate_node_name("one two three", "h2")
        assert result == "one two three"

    def test_should_return_node_type_for_empty_text_headers(self, html_parser):
        result = html_parser._generate_node_name("", "h3")
        assert result == "h3"

    def test_should_return_ul_list_for_unordered_lists(self, html_parser):
        result = html_parser._generate_node_name("some text", "ul")
        assert result == "ul-list"

    def test_should_return_ol_list_for_ordered_lists(self, html_parser):
        result = html_parser._generate_node_name("some text", "ol")
        assert result == "ol-list"

    def test_should_return_table_for_tables(self, html_parser):
        result = html_parser._generate_node_name("some text", "table")
        assert result == "table"

    def test_should_work_with_h4_h5_h6_headers(self, html_parser):
        for header_type in ["h4", "h5", "h6"]:
            result = html_parser._generate_node_name("test header", header_type)
            assert result == "test header"


class TestGetParentScope:
    """Test the _get_parent_scope helper method."""

    def test_should_return_none_for_top_level_element(self, html_parser):
        node = Mock(spec=Node)
        node.parent = None

        result = html_parser._get_parent_scope(node, b"")
        assert result is None

    def test_should_return_article_parent(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.type = "tag_name"
        tag_name.text = b"article"

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [tag_name]

        parent = Mock(spec=Node)
        parent.type = "element"
        parent.children = [start_tag]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        result = html_parser._get_parent_scope(node, b"")
        assert result == "article"

    def test_should_return_section_parent(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.type = "tag_name"
        tag_name.text = b"section"

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [tag_name]

        parent = Mock(spec=Node)
        parent.type = "element"
        parent.children = [start_tag]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        result = html_parser._get_parent_scope(node, b"")
        assert result == "section"

    def test_should_return_div_parent(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.type = "tag_name"
        tag_name.text = b"div"

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [tag_name]

        parent = Mock(spec=Node)
        parent.type = "element"
        parent.children = [start_tag]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        result = html_parser._get_parent_scope(node, b"")
        assert result == "div"

    def test_should_return_header_tag_as_parent(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.type = "tag_name"
        tag_name.text = b"h2"

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [tag_name]

        parent = Mock(spec=Node)
        parent.type = "element"
        parent.children = [start_tag]
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        result = html_parser._get_parent_scope(node, b"")
        assert result == "h2"

    def test_should_skip_non_semantic_tags(self, html_parser):
        # p tag should be skipped
        p_tag_name = Mock(spec=Node)
        p_tag_name.type = "tag_name"
        p_tag_name.text = b"p"

        p_start_tag = Mock(spec=Node)
        p_start_tag.type = "start_tag"
        p_start_tag.children = [p_tag_name]

        p_parent = Mock(spec=Node)
        p_parent.type = "element"
        p_parent.children = [p_start_tag]

        # div tag should be returned
        div_tag_name = Mock(spec=Node)
        div_tag_name.type = "tag_name"
        div_tag_name.text = b"div"

        div_start_tag = Mock(spec=Node)
        div_start_tag.type = "start_tag"
        div_start_tag.children = [div_tag_name]

        div_parent = Mock(spec=Node)
        div_parent.type = "element"
        div_parent.children = [div_start_tag]
        div_parent.parent = None

        p_parent.parent = div_parent

        node = Mock(spec=Node)
        node.parent = p_parent

        result = html_parser._get_parent_scope(node, b"")
        assert result == "div"

    def test_should_return_none_for_non_element_parent(self, html_parser):
        parent = Mock(spec=Node)
        parent.type = "document"
        parent.parent = None

        node = Mock(spec=Node)
        node.parent = parent

        result = html_parser._get_parent_scope(node, b"")
        assert result is None

    def test_should_check_all_semantic_tags(self, html_parser):
        semantic_tags = ["article", "section", "div", "main", "aside", "nav", "header", "footer"]

        for tag in semantic_tags:
            tag_name = Mock(spec=Node)
            tag_name.type = "tag_name"
            tag_name.text = tag.encode()

            start_tag = Mock(spec=Node)
            start_tag.type = "start_tag"
            start_tag.children = [tag_name]

            parent = Mock(spec=Node)
            parent.type = "element"
            parent.children = [start_tag]
            parent.parent = None

            node = Mock(spec=Node)
            node.parent = parent

            result = html_parser._get_parent_scope(node, b"")
            assert result == tag


class TestCountListItems:
    """Test the _count_list_items helper method."""

    def test_should_count_single_list_item(self, html_parser):
        li_tag = Mock(spec=Node)
        li_tag.type = "tag_name"
        li_tag.text = b"li"
        li_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [li_tag]

        li_element = Mock(spec=Node)
        li_element.type = "element"
        li_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [li_element]

        result = html_parser._count_list_items(node)
        assert result == 1

    def test_should_count_multiple_list_items(self, html_parser):
        items = []
        for _ in range(3):
            li_tag = Mock(spec=Node)
            li_tag.type = "tag_name"
            li_tag.text = b"li"
            li_tag.children = []

            start_tag = Mock(spec=Node)
            start_tag.type = "start_tag"
            start_tag.children = [li_tag]

            li_element = Mock(spec=Node)
            li_element.type = "element"
            li_element.children = [start_tag]

            items.append(li_element)

        node = Mock(spec=Node)
        node.children = items

        result = html_parser._count_list_items(node)
        assert result == 3

    def test_should_return_zero_for_empty_list(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        result = html_parser._count_list_items(node)
        assert result == 0

    def test_should_ignore_non_li_elements(self, html_parser):
        div_tag = Mock(spec=Node)
        div_tag.type = "tag_name"
        div_tag.text = b"div"
        div_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [div_tag]

        div_element = Mock(spec=Node)
        div_element.type = "element"
        div_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [div_element]

        result = html_parser._count_list_items(node)
        assert result == 0


class TestCountTableDimensions:
    """Test the _count_table_dimensions helper method."""

    def test_should_count_single_row(self, html_parser):
        # Mock a tr element
        tr_tag = Mock(spec=Node)
        tr_tag.type = "tag_name"
        tr_tag.text = b"tr"
        tr_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [tr_tag]

        tr_element = Mock(spec=Node)
        tr_element.type = "element"
        tr_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [tr_element]

        rows, cols = html_parser._count_table_dimensions(node)
        assert rows == 1

    def test_should_return_zero_for_empty_table(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        rows, cols = html_parser._count_table_dimensions(node)
        assert rows == 0
        assert cols == 0


class TestCountCellsInRow:
    """Test the _count_cells_in_row helper method."""

    def test_should_count_single_td_cell(self, html_parser):
        td_tag = Mock(spec=Node)
        td_tag.type = "tag_name"
        td_tag.text = b"td"
        td_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [td_tag]

        td_element = Mock(spec=Node)
        td_element.type = "element"
        td_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [td_element]

        result = html_parser._count_cells_in_row(node)
        assert result == 1

    def test_should_count_th_cells(self, html_parser):
        th_tag = Mock(spec=Node)
        th_tag.type = "tag_name"
        th_tag.text = b"th"
        th_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [th_tag]

        th_element = Mock(spec=Node)
        th_element.type = "element"
        th_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [th_element]

        result = html_parser._count_cells_in_row(node)
        assert result == 1

    def test_should_count_multiple_cells(self, html_parser):
        cells = []
        for tag_name in [b"td", b"th", b"td"]:
            tag = Mock(spec=Node)
            tag.type = "tag_name"
            tag.text = tag_name
            tag.children = []

            start_tag = Mock(spec=Node)
            start_tag.type = "start_tag"
            start_tag.children = [tag]

            cell_element = Mock(spec=Node)
            cell_element.type = "element"
            cell_element.children = [start_tag]

            cells.append(cell_element)

        node = Mock(spec=Node)
        node.children = cells

        result = html_parser._count_cells_in_row(node)
        assert result == 3

    def test_should_return_zero_for_empty_row(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        result = html_parser._count_cells_in_row(node)
        assert result == 0


class TestGetAttribute:
    """Test the _get_attribute helper method."""

    def test_should_extract_id_attribute(self, html_parser):
        attr_value = Mock(spec=Node)
        attr_value.type = "attribute_value"
        attr_value.text = b"my-id"

        quoted_value = Mock(spec=Node)
        quoted_value.type = "quoted_attribute_value"
        quoted_value.children = [attr_value]

        attr_name = Mock(spec=Node)
        attr_name.type = "attribute_name"
        attr_name.text = b"id"

        attribute = Mock(spec=Node)
        attribute.type = "attribute"
        attribute.children = [attr_name, quoted_value]

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [attribute]

        node = Mock(spec=Node)
        node.children = [start_tag]

        result = html_parser._get_attribute(node, "id")
        assert result == "my-id"

    def test_should_extract_class_attribute(self, html_parser):
        attr_value = Mock(spec=Node)
        attr_value.type = "attribute_value"
        attr_value.text = b"my-class other-class"

        quoted_value = Mock(spec=Node)
        quoted_value.type = "quoted_attribute_value"
        quoted_value.children = [attr_value]

        attr_name = Mock(spec=Node)
        attr_name.type = "attribute_name"
        attr_name.text = b"class"

        attribute = Mock(spec=Node)
        attribute.type = "attribute"
        attribute.children = [attr_name, quoted_value]

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [attribute]

        node = Mock(spec=Node)
        node.children = [start_tag]

        result = html_parser._get_attribute(node, "class")
        assert result == "my-class other-class"

    def test_should_return_none_for_missing_attribute(self, html_parser):
        attr_name = Mock(spec=Node)
        attr_name.type = "attribute_name"
        attr_name.text = b"class"

        attribute = Mock(spec=Node)
        attribute.type = "attribute"
        attribute.children = [attr_name]

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [attribute]

        node = Mock(spec=Node)
        node.children = [start_tag]

        result = html_parser._get_attribute(node, "id")
        assert result is None

    def test_should_return_empty_string_for_boolean_attribute(self, html_parser):
        attr_name = Mock(spec=Node)
        attr_name.type = "attribute_name"
        attr_name.text = b"disabled"

        attribute = Mock(spec=Node)
        attribute.type = "attribute"
        attribute.children = [attr_name]

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [attribute]

        node = Mock(spec=Node)
        node.children = [start_tag]

        result = html_parser._get_attribute(node, "disabled")
        assert result == ""

    def test_should_return_none_for_node_without_attributes(self, html_parser):
        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = []

        node = Mock(spec=Node)
        node.children = [start_tag]

        result = html_parser._get_attribute(node, "id")
        assert result is None


class TestGetExtra:
    """Test the _get_extra helper method."""

    def test_should_include_cleaned_text(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        result = html_parser._get_extra(node, b"", "h1", "test content")
        assert result["cleaned_text"] == "test content"

    def test_should_truncate_long_cleaned_text(self, html_parser):
        node = Mock(spec=Node)
        node.children = []
        long_text = "a" * 150

        result = html_parser._get_extra(node, b"", "h1", long_text)
        assert len(result["cleaned_text"]) == 100

    def test_should_not_include_cleaned_text_if_empty(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        result = html_parser._get_extra(node, b"", "h1", "")
        assert "cleaned_text" not in result

    def test_should_include_item_count_for_lists(self, html_parser):
        li_tag = Mock(spec=Node)
        li_tag.type = "tag_name"
        li_tag.text = b"li"
        li_tag.children = []

        start_tag = Mock(spec=Node)
        start_tag.type = "start_tag"
        start_tag.children = [li_tag]

        li_element = Mock(spec=Node)
        li_element.type = "element"
        li_element.children = [start_tag]

        node = Mock(spec=Node)
        node.children = [li_element, li_element]

        result = html_parser._get_extra(node, b"", "ul", "test")
        assert result["item_count"] == "2"

    def test_should_include_table_dimensions(self, html_parser):
        node = Mock(spec=Node)
        node.children = []

        result = html_parser._get_extra(node, b"", "table", "test")
        assert "rows" in result
        assert "cols" in result


class TestProcessMatch:
    """Test the process_match method."""

    def test_should_return_none_when_no_match(self, html_parser):
        match = {}
        result = html_parser.process_match(match, b"")
        assert result is None

    def test_should_process_h1_header(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.text = b"h1"

        text_node = Mock(spec=Node)
        text_node.type = "text"
        text_node.text = b"Welcome to the Site"
        text_node.children = []

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 28
        node.start_point = (0, 0)
        node.end_point = (0, 28)
        node.children = [text_node]
        node.parent = None

        match = {"header": [node], "tag_name": [tag_name]}
        source = b"<h1>Welcome to the Site</h1>"

        result = html_parser.process_match(match, source)
        assert result is not None
        content, node_info = result
        assert content == "<h1>Welcome to the Site</h1>"
        assert node_info["node_type"] == "h1"
        assert node_info["language"] == "html"

    def test_should_process_h2_header(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.text = b"h2"

        text_node = Mock(spec=Node)
        text_node.type = "text"
        text_node.text = b"Subheading"
        text_node.children = []

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 15
        node.start_point = (0, 0)
        node.end_point = (0, 15)
        node.children = [text_node]
        node.parent = None

        match = {"header": [node], "tag_name": [tag_name]}
        source = b"<h2>Subheading</h2>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_type"] == "h2"

    def test_should_process_table(self, html_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 20
        node.start_point = (0, 0)
        node.end_point = (2, 8)
        node.children = []
        node.parent = None

        match = {"table": [node], "tag_name": [Mock(spec=Node)]}
        source = b"<table></table>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_type"] == "table"
        assert node_info["node_name"] == "table"

    def test_should_process_unordered_list(self, html_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"ul": [node], "tag_name": [Mock(spec=Node)]}
        source = b"<ul></ul>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_type"] == "ul"
        assert node_info["node_name"] == "ul-list"

    def test_should_process_ordered_list(self, html_parser):
        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"ol": [node], "tag_name": [Mock(spec=Node)]}
        source = b"<ol></ol>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_type"] == "ol"
        assert node_info["node_name"] == "ol-list"

    def test_should_set_documentation_to_none(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.text = b"h1"

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"header": [node], "tag_name": [tag_name]}
        source = b"<h1>Test</h1>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["documentation"] is None

    def test_should_set_signature_to_none(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.text = b"h1"

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"header": [node], "tag_name": [tag_name]}
        source = b"<h1>Test</h1>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["signature"] is None

    def test_should_handle_header_with_no_tag_name_text(self, html_parser):
        tag_name = Mock(spec=Node)
        tag_name.text = None

        node = Mock(spec=Node)
        node.start_byte = 0
        node.end_byte = 10
        node.start_point = (0, 0)
        node.end_point = (0, 10)
        node.children = []
        node.parent = None

        match = {"header": [node], "tag_name": [tag_name]}
        source = b"<h1>Test</h1>"

        result = html_parser.process_match(match, source)
        assert result is not None
        _, node_info = result
        assert node_info["node_type"] == "h1"  # defaults to h1


# Integration tests


class TestParseIntegration:
    """Integration tests for the parse method with real HTML code."""

    def test_should_parse_simple_header(self, html_parser):
        content = "<h1>Hello World</h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_type == "h1"
        assert "hello" in node_metadata.node_name

    def test_should_parse_multiple_headers(self, html_parser):
        content = """
<h1>Main Title</h1>
<h2>Subtitle One</h2>
<h3>Subtitle Two</h3>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 3
        node_types = [r[1].node_type for r in results]
        assert "h1" in node_types
        assert "h2" in node_types
        assert "h3" in node_types

    def test_should_parse_unordered_list(self, html_parser):
        content = """
<ul>
    <li>Item 1</li>
    <li>Item 2</li>
    <li>Item 3</li>
</ul>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_type == "ul"
        assert node_metadata.node_name == "ul-list"
        assert node_metadata.extra["item_count"] == "3"

    def test_should_parse_ordered_list(self, html_parser):
        content = """
<ol>
    <li>First</li>
    <li>Second</li>
</ol>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_type == "ol"
        assert node_metadata.extra["item_count"] == "2"

    def test_should_parse_table(self, html_parser):
        content = """
<table>
    <tr>
        <th>Header 1</th>
        <th>Header 2</th>
    </tr>
    <tr>
        <td>Data 1</td>
        <td>Data 2</td>
    </tr>
</table>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.node_type == "table"
        assert node_metadata.extra["rows"] == "2"
        assert node_metadata.extra["cols"] == "2"

    def test_should_extract_id_attribute(self, html_parser):
        content = '<h1 id="main-title">Title</h1>'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["id"] == "main-title"

    def test_should_extract_class_attribute(self, html_parser):
        content = '<h2 class="header sub">Subtitle</h2>'
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["class"] == "header sub"

    def test_should_handle_nested_elements_in_section(self, html_parser):
        content = """
<section>
    <h1>Section Title</h1>
    <p>Some text</p>
</section>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) >= 1
        # Find the h1 result
        h1_results = [r for r in results if r[1].node_type == "h1"]
        assert len(h1_results) == 1
        assert h1_results[0][1].parent_scope == "section"

    def test_should_handle_complex_document(self, html_parser):
        content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <article>
        <h1>Main Article</h1>
        <section>
            <h2>Section One</h2>
            <ul>
                <li>Point A</li>
                <li>Point B</li>
            </ul>
        </section>
        <section>
            <h2>Section Two</h2>
            <table>
                <tr><td>Cell</td></tr>
            </table>
        </section>
    </article>
</body>
</html>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        # Should find headers, list, and table
        assert len(results) >= 4

        node_types = {r[1].node_type for r in results}
        assert "h1" in node_types
        assert "h2" in node_types
        assert "ul" in node_types
        assert "table" in node_types

    def test_should_remove_stopwords_from_headers(self, html_parser):
        content = "<h1>The quick brown fox jumps over the lazy dog</h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        # "the" and "over" should be removed
        node_name = node_metadata.node_name.lower()
        assert "quick" in node_name
        assert "brown" in node_name
        assert "fox" in node_name

    def test_should_handle_empty_file(self, html_parser):
        content = ""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=0,
            mtime=1234567890.0,
        )
        document = Document(path="empty.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 0

    def test_should_handle_file_with_no_semantic_elements(self, html_parser):
        content = """
<html>
<body>
    <p>Just a paragraph</p>
    <span>And a span</span>
</body>
</html>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 0

    def test_should_include_metadata_fields_from_document(self, html_parser):
        content = "<h1>Test</h1>"
        metadata = DocumentMetadata(
            repo="my-repo",
            repo_path="/custom/path",
            hash="hash123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="custom/page.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))

        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.repo == "my-repo"
        assert node_metadata.repo_path == "/custom/path"
        assert node_metadata.document_path == "custom/page.html"
        assert node_metadata.hash == "hash123"


class TestHtmlParserInitialization:
    """Test HtmlParser initialization and properties."""

    def test_should_initialize_successfully(self):
        parser = HtmlParser()
        assert parser.language == "html"
        assert parser.tslanguage is not None
        assert parser.tsparser is not None

    def test_should_have_query_string(self, html_parser):
        query = html_parser.query_str
        assert "header" in query
        assert "table" in query
        assert "ul" in query
        assert "ol" in query
        assert "h[1-6]" in query

    def test_should_have_stopwords(self, html_parser):
        assert len(html_parser.STOPWORDS) > 0
        assert "the" in html_parser.STOPWORDS
        assert "and" in html_parser.STOPWORDS
        assert "is" in html_parser.STOPWORDS


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_malformed_html_gracefully(self, html_parser):
        content = "<h1>Unclosed header<p>Missing closing tag"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="malformed.html", content=content, metadata=metadata)

        # Should not raise an exception
        results = list(html_parser.parse(document))
        assert isinstance(results, list)

    def test_should_handle_html_with_unicode(self, html_parser):
        content = "<h1>Hello 世界 🌍</h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="unicode.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) >= 1

    def test_should_handle_header_with_nested_tags(self, html_parser):
        content = "<h1>Hello <strong>World</strong></h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        # Should extract text from nested elements
        assert "hello" in node_metadata.node_name.lower()

    def test_should_handle_empty_list(self, html_parser):
        content = "<ul></ul>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["item_count"] == "0"

    def test_should_handle_empty_table(self, html_parser):
        content = "<table></table>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        assert node_metadata.extra["rows"] == "0"
        assert node_metadata.extra["cols"] == "0"

    def test_should_handle_header_with_only_stopwords(self, html_parser):
        content = "<h1>The and is</h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        # Should fall back to node type
        assert node_metadata.node_name == "h1"

    def test_should_handle_all_header_levels(self, html_parser):
        content = """
<h1>Level 1</h1>
<h2>Level 2</h2>
<h3>Level 3</h3>
<h4>Level 4</h4>
<h5>Level 5</h5>
<h6>Level 6</h6>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 6

        node_types = [r[1].node_type for r in results]
        for i in range(1, 7):
            assert f"h{i}" in node_types

    def test_should_handle_deeply_nested_structure(self, html_parser):
        content = """
<article>
    <section>
        <div>
            <main>
                <h1>Deep Header</h1>
            </main>
        </div>
    </section>
</article>
"""
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) >= 1
        # Should find a parent scope
        h1_results = [r for r in results if r[1].node_type == "h1"]
        assert len(h1_results) == 1
        # Should have found one of the parent scopes
        assert h1_results[0][1].parent_scope in ["article", "section", "div", "main"]

    def test_should_handle_whitespace_only_text(self, html_parser):
        content = "<h1>   \n\t  </h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        # Should fall back to h1 since no meaningful text
        assert node_metadata.node_name == "h1"

    def test_should_handle_very_long_header_text(self, html_parser):
        long_text = " ".join(["word"] * 100)
        content = f"<h1>{long_text}</h1>"
        metadata = DocumentMetadata(
            repo="test-repo",
            repo_path="/path/to/repo",
            hash="abc123",
            ext=".html",
            size_bytes=len(content),
            mtime=1234567890.0,
        )
        document = Document(path="test.html", content=content, metadata=metadata)

        results = list(html_parser.parse(document))
        assert len(results) == 1
        _, node_metadata = results[0]
        # Should only take first 5 words
        assert len(node_metadata.node_name.split()) == 5
