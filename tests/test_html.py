"""Tests for HTML parsing utilities."""

from nlp_utils.html import extract_article, extract_clean_html, html_to_markdown

_FULL_PAGE = """
<html>
<head><title>Test</title></head>
<body>
  <nav>Site navigation</nav>
  <header>Page header</header>
  <main>
    <article>
      <h1>Hello World</h1>
      <p>Article content here.</p>
    </article>
  </main>
  <aside>Related links</aside>
  <footer>Footer text</footer>
  <script>alert('xss')</script>
  <style>.foo { color: red; }</style>
</body>
</html>
"""


class TestExtractArticle:
    def test_returns_article_content(self):
        result = extract_article(_FULL_PAGE)
        assert "Hello World" in result
        assert "Article content here." in result

    def test_excludes_nav_footer(self):
        result = extract_article(_FULL_PAGE)
        assert "Site navigation" not in result
        assert "Footer text" not in result

    def test_falls_back_to_main(self):
        html = "<html><body><main><p>Main content</p></main></body></html>"
        result = extract_article(html)
        assert "Main content" in result

    def test_falls_back_to_body(self):
        html = "<html><body><p>Body content</p></body></html>"
        result = extract_article(html)
        assert "Body content" in result

    def test_empty_html(self):
        result = extract_article("<html></html>")
        assert isinstance(result, str)


class TestExtractCleanHtml:
    def test_removes_noise_tags(self):
        result = extract_clean_html(_FULL_PAGE)
        assert "Site navigation" not in result
        assert "Page header" not in result
        assert "Related links" not in result
        assert "Footer text" not in result
        assert "alert(" not in result
        assert "color: red" not in result

    def test_keeps_article_content(self):
        result = extract_clean_html(_FULL_PAGE)
        assert "Hello World" in result
        assert "Article content here." in result

    def test_iframe_removed(self):
        html = "<html><body><p>Keep</p><iframe src='x'></iframe></body></html>"
        result = extract_clean_html(html)
        assert "Keep" in result
        assert "iframe" not in result

    def test_noscript_removed(self):
        html = "<html><body><p>Keep</p><noscript>No JS</noscript></body></html>"
        result = extract_clean_html(html)
        assert "No JS" not in result


class TestHtmlToMarkdown:
    def test_headings_atx_style(self):
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        result = html_to_markdown(html)
        assert "# Title" in result
        assert "## Subtitle" in result

    def test_paragraph_text(self):
        html = "<p>Hello world</p>"
        result = html_to_markdown(html)
        assert "Hello world" in result

    def test_links_preserved(self):
        html = '<a href="https://example.com">Click here</a>'
        result = html_to_markdown(html)
        assert "Click here" in result
        assert "https://example.com" in result

    def test_strips_script_style(self):
        html = "<p>Text</p><script>bad()</script><style>.x{}</style>"
        result = html_to_markdown(html)
        assert "bad()" not in result
        assert ".x{}" not in result

    def test_returns_string(self):
        assert isinstance(html_to_markdown("<p>hi</p>"), str)
