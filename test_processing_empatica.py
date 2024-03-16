import unittest
import numpy as np
import pandas as pd
import empatica_helpers as eh


# python /teststest_processing_empatica.py
class TestProcessingEmpatica(unittest.TestCase):
    # Existing tests...

    def test_load_data_and_combine(self):
        # Assuming eh.load_data_and_combine is a function in your code
        # Replace 'input/empatica/sample_folder' with a valid test folder path
        df, trimmings, uniqueness_check = eh.load_data_and_combine('test_data/test_empatica/test_empatica_folder', verbose=False)

        # Check if the returned dataframe is not empty
        self.assertFalse(df.empty)

        # Check if trimmings is a numpy array
        self.assertIsInstance(trimmings, np.ndarray)

        # Check if uniqueness_check is a dataframe
        self.assertIsInstance(uniqueness_check, pd.DataFrame)

    def test_skip_empty_dataframe(self):
        # Assuming eh.load_data_and_combine is a function in your code
        # Replace 'input/empatica/empty_folder' with a valid test folder path that returns an empty dataframe
        df, trimmings, uniqueness_check = eh.load_data_and_combine('test_data/test_empatica/test_empatica_folder', verbose=False)

        # Check if the returned dataframe is empty
        self.assertTrue(df.empty), "Dataframe is not empty"

        # Check if trimmings is an integer
        self.assertIsInstance(trimmings, int), "Trimmings is not an integer"

    def test_append_trimmings(self):
        # Create a sample trimmings array
        trimmings_array = np.array([1, 2, 3])

        # Append a value to the trimmings array
        trimmings_array = np.append(trimmings_array, 4)

        # Check if the length of the trimmings array is correct
        self.assertEqual(len(trimmings_array), 4)

    def test_concat_dataframes(self):
        # Create sample dataframes
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'A': [4, 5, 6]})

        # Concatenate dataframes
        result = pd.concat([df1, df2])

        # Check if the concatenated dataframe has the correct shape
        self.assertEqual(result.shape, (6, 1))

if __name__ == '__main__':
    unittest.main()