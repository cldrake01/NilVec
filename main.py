import nilvec


def main():
    hnsw = nilvec.PyHNSW(128, None, None, None, None, None)
    vec = [1.0] * 128
    hnsw.insert(vec)
    print("Search")
    print(hnsw.search(vec, 1))


if __name__ == "__main__":
    main()
