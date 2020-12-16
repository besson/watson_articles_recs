class Neighbor:
    """ Representation of a user neighbor in Collaborative filtering recommendations"""

    def __init__(self, user_id: int, score: float):
        self.user_id = user_id
        self.score = score

    def __eq__(self, other):
        """
        Redefine default __eq__ (equal) conditions for this object
        INPUT
            other - other Neighbor object
        OUTPUT
            condition resolution: output of comparison, boolean
        """
        return self.user_id == other.user_id

    def __lt__(self, other):
        """
        Redefine default __lt_ (less than) conditions for this object which is needed for comparisions
        INPUT
            other - other Neighbor object
        OUTPUT
            condition resolution: output of comparison, boolean
        """
        return self.score < other.score

    def to_dict(self):
        """
            Converts object to dictionary. It is important for creating pandas DataFrame
        """
        return {
            'user_id': self.user_id,
            'score': self.score,
        }