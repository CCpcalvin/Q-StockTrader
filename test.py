from __future__ import annotations


class Pair:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
    
    def __hash__(self) -> int:
        return (self.x, self.y).__hash__()
    
    def __eq__(self, __value: Pair) -> bool:
        return self.x == __value.x and self.y == __value.y


def main():
    data = {}
    pair = Pair(1, 2)
    data[pair] = 2

    pair2 = Pair(1, 2)
    print(data[pair2])


if __name__ == "__main__":
    main()


