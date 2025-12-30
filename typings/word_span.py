from typings.span import Span


class WordSpan(Span):
    text: str

    def __eq__(self, other):
        return self.text == other.text and super(WordSpan, self).__eq__(other)

    def __str__(self):
        return f"{self.start_index = } {self.end_index = } {self.text = }"
