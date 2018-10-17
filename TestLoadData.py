import unittest
from unittest import mock

from LoadData import *
from constants import *
from test_constants import *


class TestLoadData(unittest.TestCase):
    """
    def setUp(self):
        file_mock = mock.MagicMock()
        with mock.patch('builtins.open', file_mock):
            manager = file_mock.return_value.__enter__.return_value
            manager.read.return_value = test_synopsis_content
    """

    def test_load_type(self):
        with open(SYNOPSIS_FILE) as test_syn:
            self.assertIsInstance(load_synopsis(test_syn), pd.DataFrame)

    def test_load_content_columns(self):
        with open(SYNOPSIS_FILE) as test_syn:
            synopsis = load_synopsis(test_syn)

            self.assertEqual(list(synopsis.columns), ['id', 'text'])

    def test_load_content_content(self):
        with open(SYNOPSIS_FILE) as test_syn:
            synopsis = load_synopsis(test_syn)
            self.assertEqual(list(synopsis['id'].iloc[:2]), [23890098,31186339])