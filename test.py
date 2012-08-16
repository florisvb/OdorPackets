import create_false_odor_packet as cfop
import odor_dataset as od

odor_dataset = cfop.make_false_odor_dataset()
od.prep_data(odor_dataset)
