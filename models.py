class Neighbor:

    def __init__(self, user_id: int, score: float):
        self.user_id = user_id
        self.score = score

    def __eq__(self, other):
        return self.user_id == other.user_id

    def __lt__(self, other):
        return self.score < other.score