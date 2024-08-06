class Tokenizer:
    def __init__(self, fmt: str):
        self.fmt: str = fmt
        self.pos: int = 0

    def length(self) -> int:
        return len(self.fmt)

    def substring(self, start_ind: int, length: int) -> str:
        return self.fmt[start_ind : start_ind + length]

    def peek(self, offset: int = 0) -> int | str:
        if self.pos + offset >= self.length():
            return -1
        return self.fmt[self.pos + offset]

    def peek_until(self, start_offset: int, until: int) -> int:
        offset = start_offset
        while True:
            c = self.peek(offset)
            offset += 1
            if c == -1:
                break
            if c == until:
                return offset - start_offset
        return 0

    def peek_one_of(self, offset: int, s: str) -> bool:
        return any(self.peek(offset) == c for c in s)

    def advance(self, characters: int = 1):
        self.pos = min(self.pos + characters, len(self.fmt))

    def read_one_or_more(self, c: str) -> bool:
        if self.peek() != c:
            return False

        while self.peek() == c:
            self.advance()

        return True

    def read_one_of(self, s: str) -> bool:
        if self.peek_one_of(0, s):
            self.advance()
            return True
        return False

    def read_string(self, s: str, ignore_case: bool = False) -> bool:
        if self.pos + len(s) > self.length():
            return False

        for i, _ in enumerate(s):
            c1 = s[i]
            c2 = str(self.peek(i))
            if ignore_case:
                if c1.lower() != c2.lower():
                    return False
            elif c1 != c2:
                return False

        self.advance(len(s))
        return True

    def read_enclosed(self, char_open: str, char_closed: str) -> bool:
        if self.peek() == char_open:
            length = self.peek_until(1, char_closed)
            if length > 0:
                self.advance(1 + length)
                return True

        return False
