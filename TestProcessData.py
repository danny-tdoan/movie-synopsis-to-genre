import unittest
from unittest import mock

from ProcessData import *
from constants import *
from constants_test import *


class TestProcessData(unittest.TestCase):
    def setUp(self):
        # construct dummy data for testing
        pass

    def test_get_all_genres(self):
        data = pd.DataFrame({'genres': [['a', 'b', 'c'], ['a', 'b'], ['a', 'c']]})
        all_genres = get_all_genres(data)

        self.assertEqual(len(all_genres), 3)
        self.assertSetEqual(set(all_genres), {'a', 'b', 'c'})

    def test_get_top_n_genres_type(self):
        pass

    def test_get_top_n_genres_n(self):
        pass

    def test_categorize_movie_genre_shape(self):
        pass

    def test_categorize_movie_genre_content(self):
        pass
