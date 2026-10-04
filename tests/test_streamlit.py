"""Regression for the blank page caused by an import-only Streamlit entrypoint."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

class DemoEngine:
    def __init__(self, papers, backend):
        self.papers = papers
    def search(self, query, *args, **kwargs):
        return [dict(self.papers[0], score=1, retrieval_score=1)] if query.strip() else []

@unittest.skipIf(AppTest is None, 'Install Streamlit to run UI regression checks')
class StreamlitRerunTest(unittest.TestCase):
    def test_search_and_settings_redraw_page(self):
        sys.modules.pop('paperlens.app', None)
        with patch('paperlens.engine.SearchEngine', DemoEngine):
            app = AppTest.from_file(str(Path('streamlit_app.py').resolve()), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, 'PaperLens')
            app.text_input[0].set_value('retrieval augmented generation').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, 'PaperLens')
            self.assertTrue(app.subheader)
            app.sidebar.checkbox[0].uncheck().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, 'PaperLens')
            app.text_input[0].set_value('').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, 'PaperLens')

if __name__ == '__main__':
    unittest.main()
