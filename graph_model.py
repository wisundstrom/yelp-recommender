"""This file conatins the functions for the algorithm that I implemented based on this paper:

He, Jianming, and Wesley W. Chu. "A social network-based recommender system (SNRS)."
Data mining for social network data. Springer, Boston, MA, 2010. 47-74.

"""
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


def biz_preference(driver, user_id, biz_id):
    """This function returns the estimated ratings probability distribution for a specific user and business,
    limited to ratings, users, and businesses in a specific state. The function takes the neo4j server
    connection driver object, strings of the user and business yelp ids, and the two letter state abbreviation.
    It returns an array of the estimated probability distrubtion prob(user rating of business = k)"""

    # send a cypher query to the server that returns reviews of biz by people
    # in state
    review_dist = cypher(
        driver,
        f"MATCH(rep:Reputation)<--(u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)\
        WHERE b.id='{biz_id}'\
        RETURN r.id, r.stars, u.id, collect(rep.id)",
        [
            'r.id',
            'r.stars',
            'u.id', 
            'cats'])
    
    
    # send a cypher query to the server that returns all of users reputation
    # categories
    user_categories = cypher(driver, f"\
    MATCH (u:User)-[]->(r:Reputation)\
    WHERE u.id='{user_id}' RETURN r.id", ['r.id'])

    # these manipulate the user categories and biz reviews for computation
    # later
    review_stars = review_dist['r.stars'].value_counts()
    num_reviews = review_dist['r.stars'].shape[0]
    cat_ids = list(user_categories['r.id'])
    
    # we initialize a blank list of users in the user categories
    user_in_cat = []

    for cat in cat_ids:
        # this loop sends a crypher query to retreive users in each category in
        # the state
        temp = cypher(
            driver,
            f"MATCH (s:State)<-[:REVIEWS_IN]-(u:User)-[]->(r:Reputation)\
            USING INDEX u:User(id)\
            WHERE r.id ='{cat}' AND u.id IN {list(review_dist['u.id'])} AND NOT u.id='{user_id}'\
            RETURN u.id",
            ['u.id'])
        user_in_cat.append(temp)
    
#     reviews_in_cat = []
#     for i in range(len(user_in_cat)):
#         # this loop goes through each user category and sends a cypher query to get the reviews of
#         # the business from users in the category

#         sim_user = cypher(
#             driver, f'MATCH (u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)\
#             WHERE b.id = "{biz_id}" and u.id IN {list(user_in_cat[i]["u.id"])}\
#             RETURN r.stars, u.id', ['r.stars', 'u.id'])
#         reviews_in_cat.append(sim_user)

    reviews_in_cat = []
    
    for i in range(len(user_in_cat)):
        
        sim_user = []
        
        for temp_user in user_in_cat[i]['u.id']:
            
            temp_rev=review_dist.loc[review_dist['u.id']==temp_user]
            
            sim_user.append(temp_rev['r.stars'].values[0])
            
        reviews_in_cat.append(pd.DataFrame(sim_user, columns =['r.stars']))
        
        
    # this loop and PRu below uses laplace smoothing and the distribution of biz reviews
    # to come up with naive bayes estimated probability distribution,
    # prob(review of biz = k)
    numerator = np.empty(5)
    for i in (1, 2, 3, 4, 5):
        try:
            numerator[i - 1] = review_stars[i]
        except BaseException:
            numerator[i - 1] = 0

    PRu = (numerator + 1) / (num_reviews + 5)

    # the code below uses laplace smoothing and the distribution of the biz reviews to come up with
    # a naive bayes estimate of the distribution (prob review of biz=k|given
    # reviewer in category j)
    num_cat = len(user_in_cat)
    cats_by_stars = np.empty((num_cat, 5))
    
    for i in range(num_cat):
        if not reviews_in_cat[i].empty:
            cat_stars = reviews_in_cat[i]['r.stars'].value_counts()
            #print(cat_stars)
            for j in (1, 2, 3, 4, 5):
                try:
                    cats_by_stars[i][j - 1] = cat_stars[j]
                except BaseException:
                    cats_by_stars[i][j - 1] = 0

        else:
            # If there are no users in a category we use the review
            # distribution without the conditional
            
            for j in (1, 2, 3, 4, 5):
                try:
                    cats_by_stars[i][j - 1] = review_stars[j]
                except BaseException:
                    cats_by_stars[i][j - 1] = 0

    PRaj = ((cats_by_stars + 1) / (numerator + num_cat)).prod(axis=0)

    # we now take the product of the distributions and normalize them so they
    # sum to 1
    biz_prefs_un_normalized = PRu * PRaj

    biz_prefs = biz_prefs_un_normalized / sum(biz_prefs_un_normalized)

    return biz_prefs


def user_preference(driver, user_id, biz_id):
    """This function returns the estimated ratings probability distribution for a specific user and business,
    limited to ratings, users, and businesses in the entire dataset. The function takes the driver object for
    the neo4j connection strings of the user and business yelp ids, and the two letter state abbreviation.
    It returns an array of the estimated probability distrubtion prob(user rating of business = k)"""

    # send a cypher query to the server that returns users reviews of businesses
    review_dist = cypher(driver, f"\
    MATCH (u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)-[:IN_CATEGORY]->(c:Category)\
    WHERE u.id='{user_id}' RETURN r.id, r.stars, b.id, collect(c.id)", ['r.id', 'r.stars', 'b.id', 'cats'])
    
    
    # send a cypher query to the server that returns all of the biz's
    # categories
    biz_categories = cypher(driver, f'\
    MATCH (b:Business)-[:IN_CATEGORY]->(c:Category) \
    WHERE b.id="{biz_id}" RETURN c.id', ['c.id'])

    # these manipulate the biz categories and user's reviews for computation
    # later
    review_stars = review_dist['r.stars'].value_counts()
    num_reviews = review_dist['r.stars'].shape[0]
    cat_ids = list(biz_categories['c.id'])

    # we initialize a blank list of businesses in the biz categories
    biz_in_cat = []
    for cat in cat_ids:
        # this loop sends a cypher query to retreive businesses in each
        # category in the state
        temp = cypher(driver, f'\
        MATCH (b:Business)-[:IN_CATEGORY]->(c:Category) \
        WHERE c.id="{cat}" AND b.id IN {list(review_dist["b.id"])} AND NOT b.id="{biz_id}" RETURN b.id', ['b.id'])
        biz_in_cat.append(temp)
        
    
  

   
#     for i in range(len(biz_in_cat)):
#         # this loop goes through each biz category and sends a cypher query to get the reviews of
#         # businesses in that category by the user
#         sim_biz = cypher(
#             driver, f"MATCH (u:User)-[:WROTE]->(r:Review)-[:REVIEWS]->(b:Business)\
#             WHERE u.id='{user_id}' and b.id IN {list(biz_in_cat[i]['b.id'])}\
#             RETURN r.stars, b.id", [
#                 'r.stars', 'b.id'])

#         reviews_in_cat.append(sim_biz)

    reviews_in_cat = []
    
    for i in range(len(biz_in_cat)):
        
        sim_biz = []
        
        for temp_biz in biz_in_cat[i]['b.id']:
            
            temp_rev=review_dist.loc[review_dist['b.id']==temp_biz]
            
            sim_biz.append(temp_rev['r.stars'].values[0])
            
        reviews_in_cat.append(pd.DataFrame(sim_biz, columns=['r.stars']))

    # this loop and PRu below uses laplace smoothing and the distribution of user's reviews
    # to come up with naive bayes estimated probability distribution,
    # prob(review from user = k)
    numerator = np.empty(5)
    for i in (1, 2, 3, 4, 5):
        try:
            numerator[i - 1] = review_stars[i]
        except BaseException:
            numerator[i - 1] = 0

    PRu = (numerator + 1) / (num_reviews + 5)

    # the code below uses laplace smoothing and the distribution of the biz reviews to come up with
    # a naive bayes estimate of the distribution (prob review from user =
    # k|given biz in category j)
    num_cat = len(biz_in_cat)
    cats_by_stars = np.empty((num_cat, 5))

    for i in range(num_cat):
        if not reviews_in_cat[i].empty:
            cat_stars = reviews_in_cat[i]['r.stars'].value_counts()
            
            for j in (1, 2, 3, 4, 5):
                try:
                    cats_by_stars[i][j - 1] = cat_stars[j]
                except BaseException:
                    cats_by_stars[i][j - 1] = 0
        else:
            # If there are businesses in a category we use the review
            # distribution without the conditional
            for j in (1, 2, 3, 4, 5):
                try:
                    cats_by_stars[i][j - 1] = review_stars[j]
                except BaseException:
                    cats_by_stars[i][j - 1] = 0

    PRaj = ((cats_by_stars + 1) / (numerator + num_cat)).prod(axis=0)

    # we now take the product of the distributions and normalize them so they
    # sum to 1
    user_prefs_un_normalized = PRu * PRaj

    user_prefs = user_prefs_un_normalized / sum(user_prefs_un_normalized)

    return user_prefs


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
