/**
 * mr_cache.h  —  LRU cache for ibv_mr registrations
 *
 * GPU memory registration (ibv_reg_mr with nvidia-peermem) costs ~50-200 µs.
 * We cache MRs keyed on (base_addr, length) and evict LRU entries when the
 * cache is full.  The cache is NOT thread-safe; callers must hold a lock.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <list>
#include <unordered_map>
#include <infiniband/verbs.h>

struct MRKey {
    uint64_t addr;
    size_t   len;
    bool operator==(const MRKey& o) const {
        return addr == o.addr && len == o.len;
    }
};

struct MRKeyHash {
    size_t operator()(const MRKey& k) const {
        // FNV-1a mix
        size_t h = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) {
            for (int i = 0; i < 8; i++) {
                h ^= (v & 0xff); h *= 1099511628211ULL; v >>= 8;
            }
        };
        mix(k.addr);
        mix((uint64_t)k.len);
        return h;
    }
};

class MRCache {
public:
    explicit MRCache(size_t capacity) : cap_(capacity) {}

    ~MRCache() { clear(); }

    // Returns existing MR if cached, else nullptr.
    struct ibv_mr* get(uint64_t addr, size_t len) {
        MRKey k{addr, len};
        auto it = map_.find(k);
        if (it == map_.end()) return nullptr;
        // Move to front (most-recently-used).
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        return it->second.mr;
    }

    // Insert a freshly registered MR.  Evicts LRU if at capacity.
    // Returns the evicted MR (caller must ibv_dereg_mr it), or nullptr.
    struct ibv_mr* put(uint64_t addr, size_t len, struct ibv_mr* mr) {
        struct ibv_mr* evicted = nullptr;
        if (map_.size() >= cap_) {
            // Evict least-recently-used
            MRKey lru_key = lru_.back();
            lru_.pop_back();
            auto it = map_.find(lru_key);
            evicted = it->second.mr;
            map_.erase(it);
        }
        MRKey k{addr, len};
        lru_.push_front(k);
        map_[k] = {mr, lru_.begin()};
        return evicted;
    }

    void clear() {
        // Callers are responsible for dereg; just drop pointers.
        map_.clear();
        lru_.clear();
    }

    size_t size() const { return map_.size(); }

private:
    struct Entry {
        struct ibv_mr* mr;
        std::list<MRKey>::iterator lru_it;
    };

    size_t cap_;
    std::list<MRKey> lru_;
    std::unordered_map<MRKey, Entry, MRKeyHash> map_;
};
