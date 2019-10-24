#!
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
# neo4j must be installed to talk to send queries
np.set_printoptions(suppress=True)


def cypher(driver, query, results_columns):
    """This is wrapper for sending basic cypher queries to a neo4j server. Input is a neo4j connection
    driver, a string representing a cypher queryand a list of string for data frame column names.
    returns the dataframe of the results."""

    with driver.session() as session:
        result = session.run(query)

    result_df = pd.DataFrame(result.values(), columns=results_columns)

    return result_df


def expected_rating(rating_dist):
    """this takes a distribution of probabilities by rating from one to five and returns the
    expected value of the rating"""
    runsum = 0
    for i in [1, 2, 3, 4, 5]:
        runsum += rating_dist[i - 1] * i
    return runsum


def graph_model_predict(driver, user_id, biz_id):
    """this calculates the joint probability of both the user and business based distributions
    and then returns the expected value of that distribution"""
   
    biz_pref = biz_preference(driver, user_id, biz_id)
    user_pref = user_preference(driver, user_id, biz_id)
    joint_prob = (biz_pref * user_pref) / sum(biz_pref * user_pref)
    
    return expected_rating(joint_prob)

