class DisjSet:
    def __init__(self, n):
        self.rank = [0] * n
        self.N = n
        self.parent = [-1 for i in range(n)]
        self.list = [[] for i in range(n)]
  
    def find(self, x):
          
        
        if self.parent[x] == -1:
            return -1
        
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])
  
        return self.parent[x]
  
  
    def Union(self, x, y):
        if self.parent[x] == -1 or self.parent[y] == -1:
            return
        
        xset = self.find(x)
        yset = self.find(y)
  
        if xset == yset:
            return
  
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
            self.list[yset] += self.list[xset]
  
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
            self.list[xset] += self.list[yset]
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1
            self.list[xset] += self.list[yset]
            
    def Activate(self, x):
        self.parent[x] = x
        self.rank[x] = 1
        self.list[x] = [x]
        
    def all_patterns(self):
        patterns = []
        enable = [True for i in range(self.N)]
        for i in range(self.N):
            if enable[i] == False:
                continue
            root = self.find(i)
            if root == -1:
                continue
            else:
                patterns.append(self.list[root])
                for j in self.list[root]:
                    enable[j] = False
        return patterns
    
    def misclassified_patterns(self, pattern_size):
        patterns = self.all_patterns()
        res = []
        for pattern in patterns:
            if len(pattern) >= pattern_size:
                res.append(pattern)
        return res
        