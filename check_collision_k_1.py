import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
import seaborn



# constants
SAVE_IPQS = False
SLACK = 0.1

LIST_OF_BATCHES_WITH_NON_NORMALIZED_TARGET_INPUT = [1, 2]

EPSILON_COLLISION_PROPERTY = 0.000001

NUMBER_OF_LIDAR_INPUTS = 7
FIRST_OUTPUT_VARIABLE_INDEX = 9
TOTAL_NUMBER_OF_OUTPUTS = 3

CHECK_SIMPLE_PROPERTY_NO_COLLISION_WITH_OBSTACLE_STRAIGHT_AHEAD = True
CHECK_SIMPLE_PROPERTY_WITH_SIDE = False

CREATE_GENERAL_IPQ = True

def load_data():
    """
    This function uploads the MNIST dataset, and updates the input points' dimentions.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))


    x_train = change_input_dimension(X=x_train)
    x_test = change_input_dimension(X=x_test)
    return x_train, y_train, x_test, y_test

# change dimension of input to vecotrs of the size 784

def change_input_dimension(X):
    """
    This function updates the dimention of 2 2-D input to a 1-D input.
    """
    return X.reshape((X.shape[0],X.shape[1]*X.shape[2]))


def check_model_uploaded_correctly(tf_path, last_layer_name):
    """
    This function validates that a Marabou object is uploaded correctly based on a given TensorFlow 2 path.
    """
    tf_obj = tf.saved_model.load(tf_path)
    marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputName=last_layer_name)

    _, _, x_test, y_test = load_data()
    hits = 0

    for index, single_test_sample in enumerate(x_test):
        if index < 1:
            print("index", index)
            results_with_marabou_before_softmax = np.argmax(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_test_sample])[0])
            results_without_marabou_before_softmax = np.argmax(marabou_network_obj.evaluateWithMarabou(inputValues=[single_test_sample])[0])
            results_with_tf_after_softmax = np.argmax(np.array(tf_obj([single_test_sample]))[0])
            set_of_all_results = set([results_with_marabou_before_softmax, results_without_marabou_before_softmax, results_with_tf_after_softmax])
            assert len(set_of_all_results) == 1
            largest_diff = max(abs(max(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_test_sample])[0])-max(marabou_network_obj.evaluateWithMarabou(inputValues=[single_test_sample])[0])),abs(min(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_test_sample])[0])-min(marabou_network_obj.evaluateWithMarabou(inputValues=[single_test_sample])[0])))
            assert largest_diff <= 0.00001
            if results_with_marabou_before_softmax == y_test[index]:
                hits += 1
    print("total hits: ", hits)


def check_model_uploaded_correctly_robotics(tf_path, last_layer_name):
    """
    This function checks that a DRL agent (saved in a TensorFlow format) was uploaded correctly to the Marabou verification engine.
    """
    tf_obj = tf.saved_model.load(tf_path)
    marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputName=last_layer_name)

    for index in range(1):
        single_random_input = np.random.random_sample((9,))
        print("index", index)
        results_with_marabou_before_softmax = np.argmax(marabou_network_obj.evaluateWithMarabou(inputValues=[single_random_input])[0])
        results_without_marabou_before_softmax = np.argmax(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_random_input])[0])
        results_with_tf_after_softmax = np.argmax(np.array(tf_obj([single_random_input]))[0])
        set_of_all_results = set([results_with_marabou_before_softmax, results_without_marabou_before_softmax, results_with_tf_after_softmax])
        assert len(set_of_all_results) == 1
        largest_diff = max(abs(max(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_random_input])[0])-max(marabou_network_obj.evaluateWithMarabou(inputValues=[single_random_input])[0])),abs(min(marabou_network_obj.evaluateWithoutMarabou(inputValues=[single_random_input])[0])-min(marabou_network_obj.evaluateWithMarabou(inputValues=[single_random_input])[0])))
        assert largest_diff <= 0.00001


def encode_input_perturbation(marabou_network_obj, single_sample, epsilon):
    """
    This function receives a single MNIST input and pertubes it.
    """
    for input_variable_index in range(784): # inputs perturbation of epsilon (in l_infinity norm)
        original_input_value = single_sample[input_variable_index]
        marabou_network_obj.setLowerBound(input_variable_index, max(0, original_input_value - epsilon))
        marabou_network_obj.setUpperBound(input_variable_index, min(1, original_input_value + epsilon))



def encode_runner_up_beats_current_winner(marabou_network_obj, runner_up):
    """
    This function encodes an adversarial constraint indicating that the 2nd highest output (i.e., the "runner up") receives a higher
    score than the original highest output (i.e., the "winner").
    """
    original_ru_label_mapped_to_output_index_of_specific_ensemble = FIRST_OUTPUT_VARIABLE_INDEX + runner_up
    for output_index in range(FIRST_OUTPUT_VARIABLE_INDEX, FIRST_OUTPUT_VARIABLE_INDEX+10):
        if output_index != original_ru_label_mapped_to_output_index_of_specific_ensemble:
            # eq = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq = Marabou.Equation(MarabouCore.Equation.GE)
            eq.addAddend(1, original_ru_label_mapped_to_output_index_of_specific_ensemble)
            eq.addAddend(-1, output_index)
            eq.setScalar(SLACK)
            marabou_network_obj.addEquation(eq)





def generate_single_ipq(tf_path, last_layer_name, epsilon):
    """
    This function receves a path to a TensorFlow model, and then encodes an equivalent Marabou object, which
    is subsequently saved as an "InputQuery" text file.
    """
    # marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputName=last_layer_name)
    marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputNames=last_layer_name)
    # marabou_network_obj.saveQuery("davide_before_constraints")

    _, _, x_test, y_test = load_data()
    for index, single_sample in enumerate(x_test):
        if index <= 0:
            runner_up = 0
            # true_label = y_test[index]

            # encode perturbation of the input
            encode_input_perturbation(marabou_network_obj=marabou_network_obj, single_sample=single_sample, epsilon=epsilon)

            # encode that RUNNER_UP > TRUE LABEL
            encode_runner_up_beats_current_winner(marabou_network_obj=marabou_network_obj, runner_up=runner_up)

            results = marabou_network_obj.solve('ipq.ipq')
            if len(results[1]) > 0:
                sat_assignment = list(results[1].values())[:784]
                reshaped_sat_assignment = np.reshape(sat_assignment, newshape=(28, 28))
                plt.imshow(reshaped_sat_assignment)

                plt.show()
            else:
                print("no assignment found!")



def create_general_ipq(tf_path, last_layer_name, path_to_save_ipq, non_trivial_lower_bound_on_non_cental_lidars):
    """
    This function encodes an input query text file, compatible with Marabou, and encodes trivial bounds on the inputs of the DRL agent.
    """
    # marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputName=last_layer_name)
    marabou_network_obj = Marabou.read_tf(filename=tf_path, modelType="savedModel_v2", outputNames=last_layer_name)
    # marabou_network_obj = Marabou.read_tf(filename=tf_path)

    # inputNames = ['Placeholder']
    # marabou_network_obj = Marabou.read_tf(filename=tf_path, inputNames=inputNames, outputNames=last_layer_name)

    # encode trivial bounds - ALL inputs are between [0,1]
    for input_variable in marabou_network_obj.inputVars[0][0]:
        marabou_network_obj.setLowerBound(x=input_variable, v=0)
        marabou_network_obj.setUpperBound(x=input_variable, v=1)

    # non-trivial bounds - non-central lidar
    marabou_network_obj.setLowerBound(x=0, v=non_trivial_lower_bound_on_non_cental_lidars)
    marabou_network_obj.setLowerBound(x=1, v=non_trivial_lower_bound_on_non_cental_lidars)
    marabou_network_obj.setLowerBound(x=2, v=non_trivial_lower_bound_on_non_cental_lidars)
    marabou_network_obj.setLowerBound(x=4, v=non_trivial_lower_bound_on_non_cental_lidars)
    marabou_network_obj.setLowerBound(x=5, v=non_trivial_lower_bound_on_non_cental_lidars)
    marabou_network_obj.setLowerBound(x=6, v=non_trivial_lower_bound_on_non_cental_lidars)

    # central lidar
    marabou_network_obj.setLowerBound(x=3, v=0.15)

    # distance from target
    marabou_network_obj.setLowerBound(x=8, v=0.15)

    marabou_network_obj.saveQuery(filename=path_to_save_ipq)


def encode_single_query_of_simple_property(path_to_ipq_file, upper_bound_on_central_lidar_input):
    """
    This function encodes a property checking for a single-step (k=1) collision with an obstacle straight ahead.
    """
    ensemble_marabou_network_ipq_object = Marabou.load_query(filename=path_to_ipq_file)

    LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT = list(range(NUMBER_OF_LIDAR_INPUTS))
    CENTRAL_LIDAR_INPUT_VARIABLE = int(len(LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT)/2) # variable 3
    ARRAY_OF_OUTPUT_VARIABLES = list(range(FIRST_OUTPUT_VARIABLE_INDEX, FIRST_OUTPUT_VARIABLE_INDEX + TOTAL_NUMBER_OF_OUTPUTS))
    COC_OUTPUT_VARIABLE, RIGHT_OUTPUT_VARIABLE, LEFT_OUTPUT_VARIABLE = ARRAY_OF_OUTPUT_VARIABLES
    RIGHT_AND_LEFT_OUTPUT_VARIABLES = [RIGHT_OUTPUT_VARIABLE, LEFT_OUTPUT_VARIABLE]

    # center LIDAR input represents obstacle ahead
    ensemble_marabou_network_ipq_object.setUpperBound(CENTRAL_LIDAR_INPUT_VARIABLE, upper_bound_on_central_lidar_input)

    # OUTPUT constraints - negation of wanted property
    for left_or_right_output_variable in RIGHT_AND_LEFT_OUTPUT_VARIABLES:
        # negation of wanted property results to: COC > LEFT / RIGHT
        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1, COC_OUTPUT_VARIABLE)
        eq.addAddend(-1, left_or_right_output_variable)
        eq.setScalar(SLACK)
        ensemble_marabou_network_ipq_object.addEquation(eq)

    result = Marabou.solve_query(ipq=ensemble_marabou_network_ipq_object)
    assignment_dict = result[1]
    if len(assignment_dict) == 0:
        return "unsat"
    else:
        return "sat"


def encode_single_query_of_simple_property_corner_turning(path_to_ipq_file, upper_bound_on_side_coordinate, lower_bound_on_central_coordinate, left_corner_chosen):
    """
    This function encodes a property checking for a single-step (k=1) collision with an obstacle on the side.
    """
    ensemble_marabou_network_ipq_object = Marabou.load_query(filename=path_to_ipq_file)

    LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT = list(range(NUMBER_OF_LIDAR_INPUTS))
    CENTRAL_LIDAR_INPUT_VARIABLE = int(len(LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT)/2) # variable 3
    ARRAY_OF_OUTPUT_VARIABLES = list(range(FIRST_OUTPUT_VARIABLE_INDEX, FIRST_OUTPUT_VARIABLE_INDEX + TOTAL_NUMBER_OF_OUTPUTS))
    COC_OUTPUT_VARIABLE, RIGHT_OUTPUT_VARIABLE, LEFT_OUTPUT_VARIABLE = ARRAY_OF_OUTPUT_VARIABLES
    RIGHT_AND_LEFT_OUTPUT_VARIABLES = [RIGHT_OUTPUT_VARIABLE, LEFT_OUTPUT_VARIABLE]

    # side LIDAR UB
    if left_corner_chosen:
        ensemble_marabou_network_ipq_object.setUpperBound(CENTRAL_LIDAR_INPUT_VARIABLE - 1, upper_bound_on_side_coordinate)
    else:
        ensemble_marabou_network_ipq_object.setUpperBound(CENTRAL_LIDAR_INPUT_VARIABLE + 1, upper_bound_on_side_coordinate)

    # center LIDAR LB
    ensemble_marabou_network_ipq_object.setLowerBound(CENTRAL_LIDAR_INPUT_VARIABLE, lower_bound_on_central_coordinate)

    # OUTPUT constraints - negation of wanted property
    for left_or_right_output_variable in RIGHT_AND_LEFT_OUTPUT_VARIABLES:
        # negation of wanted property results to: COC > LEFT / RIGHT
        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1, COC_OUTPUT_VARIABLE)
        eq.addAddend(-1, left_or_right_output_variable)
        eq.setScalar(SLACK)
        ensemble_marabou_network_ipq_object.addEquation(eq)

    result = Marabou.solve_query(ipq=ensemble_marabou_network_ipq_object)
    assignment_dict = result[1]
    if len(assignment_dict) == 0:
        return "unsat"
    else:
        return "sat"



def simple_property_check_no_collision_with_obstacle(path_to_general_ipq, model_name):
    """
    This function checks for various adversarial inputs which cause a violation of a single-step collision with an obstacle
    straight ahead.
    """
    # encode_single_query_of_simple_property(path_to_ipq_file=PATH_TO_GENERAL_IPQ, lower_bound_on_side_lidar_inputs=0.9, upper_bound_on_central_lidar_input=0.18)
    NORMALIZATION = 1000 
    # list of coordinates (lower_bound_on_non_central_input, upper_bound_on_central_input)
    list_of_inputs_for_which_the_property_holds = []

    # list of coordinates (lower_bound_on_non_central_input, upper_bound_on_central_input)
    list_of_inputs_for_which_the_property_does_not_hold = []

    # for lower_bound_on_non_central_input in range(NORMALIZATION+1):
    #     normalized_lower_bound_on_central_input = lower_bound_on_non_central_input / NORMALIZATION
    for upper_bound_on_central_input in range(50, 270+1):
        normalized_upper_bound_on_central_input = upper_bound_on_central_input / NORMALIZATION
        print("*"*10)
        print("UPPER BOUND ON CENTRAL LIDAR: ", normalized_upper_bound_on_central_input)

        result = encode_single_query_of_simple_property(path_to_ipq_file=path_to_general_ipq, upper_bound_on_central_lidar_input=normalized_upper_bound_on_central_input)
        if result == "unsat":
            list_of_inputs_for_which_the_property_holds.append((normalized_upper_bound_on_central_input))
        else:
            list_of_inputs_for_which_the_property_does_not_hold.append((normalized_upper_bound_on_central_input))

    # assert len(list_of_inputs_for_which_the_property_holds) + len(list_of_inputs_for_which_the_property_does_not_hold) == (NORMALIZATION + 1)* (NORMALIZATION + 1)
    print("property HOLDS: ", len(list_of_inputs_for_which_the_property_holds))
    print("property VIOLATED: ", len(list_of_inputs_for_which_the_property_does_not_hold))

    x_upper_bound_of_points_when_property_holds = [x for x in list_of_inputs_for_which_the_property_holds]
    # y_lower_bound_of_points_when_property_holds = [y for y in list_of_inputs_for_which_the_property_holds]

    x_upper_bound_of_points_when_property_does_not_hold = [x for x in list_of_inputs_for_which_the_property_does_not_hold]
    # y_lower_bound_of_points_when_property_does_not_hold = [y for y in list_of_inputs_for_which_the_property_does_not_hold]

    points_with_property_holding = plt.scatter(x=x_upper_bound_of_points_when_property_holds, y=[0 for i in range(len(x_upper_bound_of_points_when_property_holds))], marker='o', label='a')
    points_with_property_not_holding = plt.scatter(x=x_upper_bound_of_points_when_property_does_not_hold, y=[1 for i in range(len(x_upper_bound_of_points_when_property_does_not_hold))], marker='o', c="darkred")
    # plt.style.use('seaborn-whitegrid')
    plt.title("simple property: robot doesn't collide with direct object\n"+ model_name)
    plt.xlabel("upper bound on central coordinate")
    # plt.ylabel("lower bound on non-central coordinates")

    plt.legend((points_with_property_holding, points_with_property_not_holding), ("property holds", "property violated"))
    plt.show()


def check_no_collision_with_side_obstacle(path_to_general_ipq, left_corner_chosen):
    """
    This function checks for various adversarial inputs which cause a violation of a single-step collision with an obstacle
    on the side.
    """
    NORMALIZATION = 100
    list_of_inputs_for_which_the_property_holds, list_of_inputs_for_which_the_property_does_not_hold = [], []
    for upper_bound_on_side_coordinates_with_corner in range(5, 30+1):
        normalized_upper_bound_on_side_and_central_coordinates_with_corner = upper_bound_on_side_coordinates_with_corner / NORMALIZATION
        for lower_bound_on_central_coordinate in range(5, 30+1): #[0.141]:
            normalized_lower_bound_on_central_coordinate = lower_bound_on_central_coordinate / 100
            result = encode_single_query_of_simple_property_corner_turning(path_to_ipq_file=path_to_general_ipq, upper_bound_on_side_coordinate=normalized_upper_bound_on_side_and_central_coordinates_with_corner, lower_bound_on_central_coordinate=normalized_lower_bound_on_central_coordinate, left_corner_chosen=left_corner_chosen)

            if result == "unsat":
                list_of_inputs_for_which_the_property_holds.append((normalized_upper_bound_on_side_and_central_coordinates_with_corner, normalized_lower_bound_on_central_coordinate))
            else:
                list_of_inputs_for_which_the_property_does_not_hold.append((normalized_upper_bound_on_side_and_central_coordinates_with_corner, normalized_lower_bound_on_central_coordinate))

    print("property HOLDS: ", len(list_of_inputs_for_which_the_property_holds))
    print("property VIOLATED: ", len(list_of_inputs_for_which_the_property_does_not_hold))

    x_upper_bound_of_points_when_property_holds = [tup[0] for tup in list_of_inputs_for_which_the_property_holds]
    y_lower_bound_of_points_when_property_holds = [tup[1] for tup in list_of_inputs_for_which_the_property_holds]

    x_upper_bound_of_points_when_property_does_not_hold = [tup[0] for tup  in list_of_inputs_for_which_the_property_does_not_hold]
    y_lower_bound_of_points_when_property_does_not_hold = [tup[1] for tup in list_of_inputs_for_which_the_property_does_not_hold]

    points_with_property_holding = plt.scatter(x=x_upper_bound_of_points_when_property_holds, y=y_lower_bound_of_points_when_property_holds, marker='o', label='a')
    points_with_property_not_holding = plt.scatter(x=x_upper_bound_of_points_when_property_does_not_hold, y=y_lower_bound_of_points_when_property_does_not_hold, marker='o', c="darkred")
    # plt.style.use('seaborn-whitegrid')
    string_of_side = "LEFT" if left_corner_chosen else "RIGHT"
    plt.title("simple property: robot doesn't collide with " + string_of_side + " side object")
    plt.xlabel("upper bound on SIDE coordinate")
    plt.ylabel("lower bound on CENTRAL coordinates")

    plt.legend((points_with_property_holding, points_with_property_not_holding), ("property holds", "property violated"))
    plt.show()


def analyze_parsed_cluster_outputs_on_corner_collision_property(path_to_csv_of_parsed_cluster_outputs):
    """
    This function analyzes a parsed csv with raw verification results on collision with corner-side obstacles.
    """
    df = pd.read_csv(path_to_csv_of_parsed_cluster_outputs)
    corner = df["corner"].unique()[0]
    assert len(df["corner"].unique()) == 1

    x_upper_bound_of_points_when_property_holds = list(df[df["ipq result"]=="unsat"]["upper_bound_on_corner_coordinates"])
    y_lower_bound_of_points_when_property_holds = list(df[df["ipq result"]=="unsat"]["lower_bound_on_non_corner_coordinates"])

    x_upper_bound_of_points_when_property_violated = list(df[df["ipq result"]=="sat"]["upper_bound_on_corner_coordinates"])
    y_lower_bound_of_points_when_property_violated = list(df[df["ipq result"]=="sat"]["lower_bound_on_non_corner_coordinates"])


    points_with_property_holding = plt.scatter(x=x_upper_bound_of_points_when_property_holds, y=y_lower_bound_of_points_when_property_holds, marker='o', label='a')
    points_with_property_not_holding = plt.scatter(x=x_upper_bound_of_points_when_property_violated, y=y_lower_bound_of_points_when_property_violated, marker='o', c="darkred")
    # plt.style.use('seaborn-whitegrid')
    plt.title("simple property: robot doesn't turn and collide with wall on "+ corner+" side")
    plt.xlabel("upper bound on coordinates on side of a wall")
    plt.ylabel("lower bound on coordinates oin side without a wall")

    # plt.legend((points_with_property_holding, points_with_property_not_holding), ("property holds", "property violated"))
    plt.show()
    pass


def simple_property_check_no_collision_with_obstacle_for_cluster(marabou_network_obj, collision_lidar_index, collision_lidar_lower_bound, collision_lidar_upper_bound, output_slack, batch_number):
    """
    This function analyzes a parsed csv with raw verification results on collisions with straight-ahead obstacles.
    """
    ALL_INPUTS = marabou_network_obj.inputVars[0][0]
    LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT = ALL_INPUTS[:7]
    ANGLE_INPUT, DISTANCE_INPUT = ALL_INPUTS[7:]
    CENTRAL_LIDAR_INPUT_VARIABLE = int(len(LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT)/2) # variable 3

    ARRAY_OF_OUTPUT_VARIABLES = marabou_network_obj.outputVars[0]
    COC_OUTPUT_VARIABLE, RIGHT_AND_LEFT_OUTPUT_VARIABLES = ARRAY_OF_OUTPUT_VARIABLES[0], ARRAY_OF_OUTPUT_VARIABLES[1:3]


    for lidar_input in LIDAR_INPUT_VARIABLES_FROM_LEFT_TO_RIGHT:
        if lidar_input != collision_lidar_index:
            marabou_network_obj.setLowerBound(x=lidar_input, v=0.2)
            marabou_network_obj.setUpperBound(x=lidar_input, v=1)

    # angle input with trivial bounds - [0,1]
    marabou_network_obj.setLowerBound(x=ANGLE_INPUT, v=0)
    marabou_network_obj.setUpperBound(x=ANGLE_INPUT, v=1)

    # distance input with bounds - [0.2, 1]
    marabou_network_obj.setLowerBound(x=DISTANCE_INPUT, v=0.2)
    marabou_network_obj.setUpperBound(x=DISTANCE_INPUT, v=1.5 if batch_number in LIST_OF_BATCHES_WITH_NON_NORMALIZED_TARGET_INPUT else 1) 

    # center LIDAR input represents obstacle ahead
    marabou_network_obj.setLowerBound(x=collision_lidar_index, v=collision_lidar_lower_bound)
    marabou_network_obj.setUpperBound(x=collision_lidar_index, v=collision_lidar_upper_bound - EPSILON_COLLISION_PROPERTY)


    # OUTPUT constraints - negation of wanted property
    for left_or_right_output_variable in RIGHT_AND_LEFT_OUTPUT_VARIABLES:
        # negation of wanted property results to: COC > LEFT / RIGHT
        eq = Marabou.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1, COC_OUTPUT_VARIABLE)
        eq.addAddend(-1, left_or_right_output_variable)
        eq.setScalar(output_slack)
        marabou_network_obj.addEquation(eq)








# create_general_ipq('/Users/wen-chungcheng/Verifying-Learning-Based-Robotic-Navigation-Systems--Artifact/models/batch_1/DDQN_id47_ep26095/saved_model.pb', 'y_out', '/Users/wen-chungcheng/Marabou_with_mods/ipqs', 0.2)
# create_general_ipq('saved_model.pb', 'y_out', '/Users/wen-chungcheng/Marabou_with_mods/ipqs', 0.2)
# create_general_ipq('/Users/wen-chungcheng/Verifying-Learning-Based-Robotic-Navigation-Systems--Artifact/models/batch_1/DDQN_id47_ep26095/saved_model.pb', 'y_out', '/Users/wen-chungcheng/Marabou_with_mods/ipqs', 0.2)
# create_general_ipq('/Users/wen-chungcheng/Verifying-Learning-Based-Robotic-Navigation-Systems--Artifact/models/batch_1/DDQN_id47_ep26095/', 'Func/StatefulPartitionedCall/input/_0', '/Users/wen-chungcheng/Marabou_with_mods/ipqs/test_ipq_0.ipq', 0.2)
# generate_single_ipq('/Users/wen-chungcheng/Verifying-Learning-Based-Robotic-Navigation-Systems--Artifact/models/batch_1/DDQN_id47_ep26095/', 'Func/StatefulPartitionedCall/input/_0', 0.2)
# encode_single_query_of_simple_property('/Users/wen-chungcheng/Marabou_with_mods/ipqs/test_ipq_0.ipq', 1)

create_general_ipq('/Users/wen-chungcheng/Verifying-Learning-Based-Robotic-Navigation-Systems--Artifact/models/batch_1/DDQN_id47_ep26095/', 'Func/StatefulPartitionedCall/input/_0', '/Users/wen-chungcheng/Marabou_with_mods/ipqs/test_ipq_0.ipq', 0.2)

simple_property_check_no_collision_with_obstacle('/Users/wen-chungcheng/Marabou_with_mods/ipqs/test_ipq_0.ipq', 'model_name')
