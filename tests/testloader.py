import unittest
import os
from src.csv_object_list_dataset_loader.loader import Loader, Scenario, Entity, EntityState


class TestTaf(unittest.TestCase):
    """Test case for a taf dataset."""

    def setUp(self):
        file_dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.loader_inst = Loader()
        self.csv_path = file_dir + "/data/taf/vehicle_tracks_000.csv"
        self.pdata_path = file_dir + "/data/taf/vehicle_tracks_000.pdata"
        self.loader_inst.load_dataset(self.csv_path)

    def test_scenario_creation_with_correct_data(self):
        self.assertIsInstance(self.loader_inst.return_scenario(self.csv_path), Scenario)

    def test_pdata_creation(self):
        self.assertTrue(os.path.exists(self.pdata_path))

    def tearDown(self):
        if os.path.exists(self.pdata_path):
            os.remove(self.pdata_path)


class TestInd(unittest.TestCase):
    """Test case for a inD dataset."""

    def setUp(self):
        file_dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.loader_inst = Loader()
        self.csv_path = file_dir + "/data/inD/01_tracks.csv"
        self.pdata_path = file_dir + "/data/inD/01_tracks.pdata"
        self.loader_inst.load_dataset(self.csv_path)

    def test_scenario_creation_with_correct_data(self):
        self.assertIsInstance(self.loader_inst.return_scenario(self.csv_path), Scenario)

    def test_scenario_identification(self):
        self.assertEqual(self.loader_inst.return_scenario(self.csv_path).id, "01")

    def test_pdata_creation(self):
        self.assertTrue(os.path.exists(self.pdata_path))

    def tearDown(self):
        if os.path.exists(self.pdata_path):
            os.remove(self.pdata_path)


class TestInteraction(unittest.TestCase):
    """Test case for a Interaction v1.2 dataset."""

    def setUp(self):
        file_dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.loader_inst = Loader()
        self.csv_path = file_dir + "/data/interaction_v12/01_tracks.csv"
        self.pdata_path = file_dir + "/data/interaction_v12/01_tracks.pdata"
        self.loader_inst.load_dataset(self.csv_path)

    def test_scenario_creation_with_correct_data(self):
        self.assertIsInstance(self.loader_inst.return_scenario(
            self.csv_path, case_id="1.0"), Scenario)

    def test_scenario_identification(self):
        self.assertEqual(self.loader_inst.return_scenario(self.csv_path, case_id="1.0").id, "1.0")
        self.assertEqual(self.loader_inst.return_scenario(self.csv_path, case_id="2.0").id, "2.0")

    def test_pdata_creation(self):
        self.assertTrue(os.path.exists(self.pdata_path))

    def tearDown(self):
        if os.path.exists(self.pdata_path):
            os.remove(self.pdata_path)


class TestBadPath(unittest.TestCase):
    def test_for_non_existent_path(self):
        """Test case for non-existent path."""
        pass


class TestEntityInD(unittest.TestCase):
    def setUp(self):
        file_dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.loader_inst = Loader()
        self.csv_path = file_dir + "/data/inD/01_tracks.csv"
        self.pdata_path = file_dir + "/data/inD/01_tracks.pdata"
        self.loader_inst.load_dataset(self.csv_path)
        self.scenario = self.loader_inst.return_scenario(self.csv_path)
        self.loaded_entity_3 = self.scenario.get_entity(3)  # pedestrian
        self.loaded_entity_4 = self.scenario.get_entity(4)  # car

    def test_state_properties(self):
        self.assertIsInstance(self.loaded_entity_3, Entity)
        self.assertEqual(self.loaded_entity_4.entity_id, 4)
        self.assertAlmostEqual(self.loaded_entity_4.length, -43.56348, places=4)
        self.assertAlmostEqual(self.loaded_entity_4.width, 189.44065, places=4)
        self.assertEqual(self.loaded_entity_4.classification, "car")
        self.assertEqual(len(self.loaded_entity_4.all_entity_states), 7)
        self.assertEqual(self.loaded_entity_4.get_all_entity_states_as_time_series().shape[0], 7)
        self.assertEqual(self.loaded_entity_4.get_all_entity_states_as_time_series().shape[1], 11)

        self.assertIsInstance(self.loaded_entity_3.get_entity_state(40), EntityState)
        entity_state = self.loaded_entity_3.get_entity_state(40)
        self.assertAlmostEqual(entity_state.x, 145.46727, places=4)
        self.assertAlmostEqual(entity_state.y, -22.24394, places=4)
        self.assertAlmostEqual(entity_state.vx, 0.0, places=4)
        self.assertAlmostEqual(entity_state.vy, 0.0, places=4)
        self.assertAlmostEqual(entity_state.ax, -1.61431, places=4)
        self.assertAlmostEqual(entity_state.ay, 0.95227, places=4)
        self.assertAlmostEqual(entity_state.yaw, -22.24394 / 180 * 3.14159262, places=4)
        self.assertAlmostEqual(entity_state.vel, 0.0, places=4)
        self.assertEqual(entity_state.timestamp, 40)
        self.assertEqual(entity_state.frame_id, 1)
        entity_state._frame_id = 14
        self.assertEqual(entity_state.frame_id, 14)

        self.assertIsInstance(self.loaded_entity_3, Entity)
        self.assertEqual(self.loaded_entity_3.entity_id, 3)
        self.assertAlmostEqual(self.loaded_entity_3.length, -22.03159, places=4)
        self.assertAlmostEqual(self.loaded_entity_3.width, 141.47493, places=4)
        self.assertEqual(self.loaded_entity_3.classification, "pedestrian")
        self.assertEqual(len(self.loaded_entity_3.all_entity_states), 7)
        self.assertEqual(self.loaded_entity_4.get_all_entity_states_as_time_series().shape[0],
                         7)
        self.assertEqual(self.loaded_entity_4.get_all_entity_states_as_time_series().shape[1],
                         11)

    def tearDown(self):
        if os.path.exists(self.pdata_path):
            os.remove(self.pdata_path)


class TestEntityInter(unittest.TestCase):
    def setUp(self):
        file_dir = str(os.path.dirname(os.path.realpath(__file__)))
        self.loader_inst = Loader()
        self.csv_path = file_dir + "/data/interaction_v12/01_tracks.csv"
        self.pdata_path = file_dir + "/data/interaction_v12/01_tracks.pdata"
        self.loader_inst.load_dataset(self.csv_path)
        self.scenario_1 = self.loader_inst.return_scenario(self.csv_path, case_id="1.0")
        self.scenario_35 = self.loader_inst.return_scenario(self.csv_path, case_id="35.0")
        self.loaded_entity_1_1 = self.scenario_1.get_entity(1)  # car
        self.loaded_entity_35_1 = self.scenario_35.get_entity(1)  # bicyclist

    def test_state_properties(self):
        self.assertIsInstance(self.loaded_entity_1_1, Entity)
        self.assertIsInstance(self.loaded_entity_35_1, Entity)
        self.assertEqual(self.loaded_entity_35_1.entity_id, 1)
        self.assertEqual(self.loaded_entity_35_1.length, 0.0)
        self.assertEqual(self.loaded_entity_35_1.classification, "pedestrian/bicycle")
        self.assertEqual(len(self.loaded_entity_35_1.all_entity_states), 40)
        self.assertEqual(
            self.loaded_entity_35_1.get_all_entity_states_as_time_series().shape[0], 40)
        self.assertEqual(
            self.loaded_entity_35_1.get_all_entity_states_as_time_series().shape[1], 11)

        self.assertIsInstance(self.loaded_entity_35_1.get_entity_state(3600), EntityState)
        entity_state = self.loaded_entity_35_1.get_entity_state(3600)
        self.assertAlmostEqual(entity_state.x, 1011.717, places=3)
        self.assertAlmostEqual(entity_state.y, 958.266, places=3)
        self.assertAlmostEqual(entity_state.vx, -1.357, places=3)
        self.assertAlmostEqual(entity_state.vy, -0.452, places=3)
        self.assertEqual(entity_state.yaw, 0.0)
        self.assertAlmostEqual(entity_state.vel, 1.4302, places=3)
        self.assertEqual(entity_state.timestamp, 3600)
        self.assertEqual(entity_state.frame_id, 36)
        entity_state._frame_id = 14
        self.assertEqual(entity_state.frame_id, 14)

    def tearDown(self):
        if os.path.exists(self.pdata_path):
            os.remove(self.pdata_path)


class TestBadDataset(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
