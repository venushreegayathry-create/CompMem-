"""Phase 2: Multi-Index Hashing (MIH) with multi-probe search + medoid clustering."""
from collections import defaultdict
from typing import Dict, List, Tuple

N_BITS, S = 64, 4
PIECE_BITS = N_BITS // S        # 16
PIECE_MASK = (1 << PIECE_BITS) - 1


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def pieces(code: int) -> List[int]:
    return [(code >> (PIECE_BITS * k)) & PIECE_MASK for k in range(S)]


class MIHIndex:
    """4 hash tables, one per 16-bit piece.
    Pigeonhole: Hamming(a,b) <= r  =>  some piece differs by <= floor(r/S) bits.
    For r=7 -> <=1 bit, so probing the exact piece + its 16 one-bit flips is exact."""

    def __init__(self):
        self.tables: List[Dict[int, set]] = [defaultdict(set) for _ in range(S)]
        self.codes: Dict[int, int] = {}          # id -> 64-bit code

    def add(self, item_id: int, code: int):
        self.codes[item_id] = code
        for k, p in enumerate(pieces(code)):
            self.tables[k][p].add(item_id)

    def remove(self, item_id: int):
        code = self.codes.pop(item_id)
        for k, p in enumerate(pieces(code)):
            self.tables[k][p].discard(item_id)

    def search(self, q: int, r: int = 7) -> List[Tuple[int, int]]:
        """Return [(item_id, hamming_dist)] with dist <= r, sorted by distance."""
        flips = r // S                            # max bit flips per piece
        assert flips <= 1, "this implementation probes up to 1 flipped bit per piece"
        cand = set()
        for k, p in enumerate(pieces(q)):
            cand |= self.tables[k].get(p, set())                   # exact piece
            if flips >= 1:
                for b in range(PIECE_BITS):                        # 16 neighbours
                    cand |= self.tables[k].get(p ^ (1 << b), set())
        res = [(i, hamming(q, self.codes[i])) for i in cand]
        return sorted([x for x in res if x[1] <= r], key=lambda t: t[1])

    def brute_force(self, q: int, r: int = 7):
        res = [(i, hamming(q, c)) for i, c in self.codes.items()]
        return sorted([x for x in res if x[1] <= r], key=lambda t: t[1])


class Cluster:
    def __init__(self, cid, medoid_id, medoid_code):
        self.id, self.medoid_id, self.medoid_code = cid, medoid_id, medoid_code
        self.members: List[int] = []
        self.bit_sum = [0] * N_BITS          # running sum of +-1 per bit

    def add(self, item_id, code):
        self.members.append(item_id)
        for i in range(N_BITS):
            self.bit_sum[i] += 1 if (code >> i) & 1 else -1

    @property
    def mu(self):
        """Average barcode mu_j in [-1,1]^64."""
        n = max(len(self.members), 1)
        return [s / n for s in self.bit_sum]


class MemoryStore:
    """Stores chunks in the MIH index and groups near-duplicates into clusters B_j."""

    def __init__(self, cluster_radius: int = 4, search_radius: int = 7):
        self.index = MIHIndex()                 # all chunks
        self.medoid_index = MIHIndex()          # cluster medoids only (fast assignment)
        self.clusters: Dict[int, Cluster] = {}
        self.chunk_cluster: Dict[int, int] = {}
        self.texts: Dict[int, str] = {}
        self.cluster_radius, self.search_radius = cluster_radius, search_radius
        self._next_id = 0
        self._next_cluster = 0                  # monotonic: ids must never be reused after deletions

    def save(self, text: str, code: int) -> int:
        cid_chunk = self._next_id; self._next_id += 1
        self.texts[cid_chunk] = text
        self.index.add(cid_chunk, code)
        hits = self.medoid_index.search(code, r=self.cluster_radius)
        if hits:                                # within 4 bits of a medoid -> join
            cl = self.clusters[hits[0][0]]
        else:                                   # else this chunk starts a new cluster
            cl = Cluster(self._next_cluster, cid_chunk, code)
            self._next_cluster += 1
            self.clusters[cl.id] = cl
            self.medoid_index.add(cl.id, code)
        cl.add(cid_chunk, code)
        self.chunk_cluster[cid_chunk] = cl.id
        return cid_chunk

    def query(self, code: int, r: int = None):
        r = self.search_radius if r is None else r
        return [(i, d, self.texts[i], self.chunk_cluster[i]) for i, d in self.index.search(code, r)]
