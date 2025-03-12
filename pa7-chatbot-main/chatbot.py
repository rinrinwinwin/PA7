# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import os.path

import util
from pydantic import BaseModel, Field
import porter_stemmer
import random
import numpy as np
import re
import nltk

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'OurBot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.user_ratings = np.zeros(len(self.titles))

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################
        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings, threshold=2.5)
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hello! I am a ChatBot that can recommend movies to you based on your watch history. To start, please tell me about some movies you have watched. (Enter ':quit' to stop at any time.)"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)

        else:
            # need index and rating of each movie; min of 5 movies to get recommendation
            rec_keywords = r"\b(?:un)?recommend(?:s|ed|ing|ation|ations)?\b"
            line = self.preprocess(line)
            titles = self.extract_titles(line)
            if titles: # when a movie title is found; processes one at a time
                if len(titles) == 1: # if only one relevant match
                    matches = self.find_movies_by_title(titles[0])
                    if len(matches) == 1:
                        sentiment = self.extract_sentiment(line)
                        if sentiment == 1:
                            response = 'I see that you liked ' + titles[0] + '!'
                            self.user_ratings[matches[0]] = 1
                        if sentiment == 0:
                            response = 'I see that you watched ' + titles[0] + '! Can you tell what you thought about it?'
                        if sentiment == -1:
                            response = 'I see that you disliked ' + titles[0] + '!'
                            self.user_ratings[matches[0]] = -1
                    elif len(titles) > 1:
                        response = 'I found multiple matches for your movie! Can you specify the year in which your movie was produced and tell me about it again? ((e.g. "Titanic (1997)" was good.)'
                    else:
                        response = "Hmm...I couldn't find any matching movies in my database. Could you tell me about another movie or check if you made a typo?"
                else:
                    response = 'Sorry, I can only process one movie at a time. Tell me more about "' + titles[0] + '".'
            elif bool(re.search(rec_keywords, line)):
                if np.count_nonzero(self.user_ratings) >= 5:
                    recommendation = self.recommend(self.user_ratings, self.ratings, 1)
                    response = str(recommendation)
                else:
                    response = "I need to hear more about at least 5 movies before I can recommend anything. You have told me about " + str(np.count_nonzero(self.user_ratings)) + " so far."
            else:
                response = 'Sorry, I did not detect a movie title in your response. Please make sure your movie title is in quotation marks and the year (if applicable) is in parentheses. (e.g. "Titanic (1997)").'


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################
        text = text.rstrip()
        for p in ["?", ",", ".", "-", "_", "—", "&", "!"]:
            text = text.replace(p, "")
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def parse_out_movies(self, s):
        """Extract movie titles enclosed in quotation marks from text.
        
        This function parses through the input string and identifies any text
        enclosed within quotation marks ("), assuming these are movie titles.
        
        :param s: A string possibly containing movie titles in quotation marks
        :returns: A tuple containing (list of extracted movie titles, remaining text)
                - The first element is a list of strings found within quotation marks
                - The second element is the input string with all quoted text removed
        """
        res = []
        track = False
        curr = ""
        full = ""
        for x in s:
            if x in ['"']:
                track = not(track)
                if not(track):
                    res.append(curr)
                    curr = ""
            else:
                if track:
                    curr += x
                else:
                    full += x
        return res, full
    

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = r'"([^"]+)"'
        titles = re.findall(pattern, preprocessed_input)
        return titles


    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        def process_features(text):
            text = text.lower()
            year_match = re.search(r'\(\d{4}\)', text)
            year = year_match.group(0) if year_match else None
            if year:
                text = text.replace(year, "")
                
            for p in ["?", ",", ".", "-", "_", "—", "&", "!"]:
                text = text.replace(p, "")
            
            stopwords = {"i", "an", "a", "the"}
            features = [word.strip() for word in text.split() if word.strip() not in stopwords]
            return features, year

        def is_match(query_features, movie_features):
            query_words, query_year = query_features
            movie_words, movie_year = movie_features
            return (query_words == movie_words) and (query_year is None or movie_year is None or query_year == movie_year)
        
        def find_matches(t):
            query_features = process_features(t)
            return [
                idx for idx, movie in enumerate(self.titles)
                if is_match(query_features, process_features(movie[0]))
            ]
            
        return find_matches(use_title)


    def find_movie_by_index(self, index):
        pass

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        PS = nltk.stem.PorterStemmer()
        def conv_to_value(sent):
            if sent is None:
                return 0
            elif sent == "pos":
                return 1
            else:
                return -1
                
        def sentiment(w):
            sent = None
            if w in self.sentiment:
                sent = self.sentiment[w]
            else:
                w_stemmed = PS.stem(w)
                if w_stemmed in self.sentiment:
                    sent = self.sentiment[w_stemmed]
            return conv_to_value(sent)

        opps = ["not", "can't", "never", "didn't", "wasn't", "shouldn't", "shan't", "don't"]
        mults = ["extremely", "terrifyingly", "astronomically", "really", "super", "overwhelmingly", "extravaggently", "excrutiatingly", "totally", "so", "very", "juicy"]

        movie_titles, sentence = self.parse_out_movies(preprocessed_input)
        words =[x.strip() for x in sentence.lower().split()]

        res = 0
        running = 1
        for w in words:
            #print(w, sentiment(w), PS.stem("enjoyed"))
            if w in mults:
                running *= 2
            elif w in opps:
                running *= -1
            else:
                res += running * sentiment(w)
            
        return np.clip(res, -1, 1)


    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################
        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.where(ratings == 0, 0, np.where(ratings > threshold, 1, -1))
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        def norm(x):
            return np.sqrt(np.sum(np.square(x)))
        denom = norm(u) * norm(v)
        numer = np.sum(u * v)
        if denom == 0:
            similarity = 0
        else:
            similarity = numer/denom
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.

        recommendations = []

        if not llm_enabled:
            rated_movies = np.nonzero(user_ratings)[0]  # get all movies user has rated as list of indices
            cos_sum = np.zeros((ratings_matrix.shape[0],))
            norms = np.linalg.norm(ratings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            normalized_ratings = ratings_matrix / norms
            for index in rated_movies:  # for every movie the user has rated
                target_row = normalized_ratings[index]  # get all user ratings for target movie
                similarity_matrix = normalized_ratings.dot(target_row)  # get cosine similarity between target movie and all unwatched movies
                contribution = user_ratings[index] * similarity_matrix  # multiply similarity matrix by rating
                cos_sum += contribution  # summation
            cos_sum[rated_movies] = -np.inf #eliminates movies user has seen from contention
            recommendations = np.argsort(cos_sum)[-k:][::-1].tolist()

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
