"""
Simple LRU cache for storing computed values.
"""


class Cache:
    def __init__(self, max_size=100000):
        """
        Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache = {}
    
    def get(self, key):
        """Get value from cache, returns None if not found."""
        return self.cache.get(key)
    
    def put(self, key, value):
        """Put value into cache."""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove a random item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def __contains__(self, key):
        return key in self.cache
    
    def __len__(self):
        return len(self.cache)
