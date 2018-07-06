from logging.config import fileConfig
from unittest import TestCase
from ddt import ddt, data,unpack
import os

from LoadLargeDataset import load_large_dataset


@ddt
class TestSitLoadLargeDataset(TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("data", 6))
    @unpack
    def test_load_large_dataset(self, base_dir, expected_total_lines):
        #Arrange
        base_dir_full_path = os.path.join(os.path.dirname(__file__), base_dir)

        #Arrange
        actual = load_large_dataset(base_dir_full_path)

        #Assert
        actual_list = [r for r in actual]
        self.assertEqual(expected_total_lines, len(actual_list))
