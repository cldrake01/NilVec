const std = @import("std");
// const set = @import("ziglangSet");
const testing = std.testing;

const ID = @import("id.zig").ID;
const Metadata = @import("metadata.zig").Metadata;
const Candidate = @import("candidate.zig").Candidate;
const Metric = @import("metric.zig").Metric;
const Filter = @import("filter.zig").Filter;
const MinMaxHeap = @import("min_max_heap.zig").MinMaxHeap;

// Inserts a Candidate into a sorted ArrayList while maintaining order
fn insort(nns: *std.ArrayList(Candidate), item: Candidate) !void {
    var i: usize = 0;

    // Find the correct insertion index based on distance
    while (i < nns.items.len and nns.items[i].distance < item.distance) {
        i += 1;
    }

    // Insert at index i by shifting elements right
    try nns.insert(i, item);
}

pub fn HNSW(comptime T: type, comptime N: usize) type {
    comptime {
        switch (@typeInfo(T)) {
            .Int, .Float, .ComptimeInt, .ComptimeFloat => {},
            else => {
                @compileError("HNSW only supports integer and floating point types.");
            },
        }

        if (N == 0) {
            @compileError("Dimension cannot be zero.");
        }
    }

    return struct {
        const Self = @This();

        // Number of layers
        layers: usize,
        // Maximum number of neighbors
        m: usize,
        ml: f64,
        // Number of neighbors to consider during construction
        ef_construction: usize,
        // Number of neighbors to consider during search
        ef_search: usize,
        vectors: std.ArrayList([N]T),
        // connections: std.ArrayList(std.ArrayList(ID)),
        connections: std.ArrayList(ID),
        // Start index of neighbors for each node
        offsets: std.ArrayList(usize),
        // Level of each node
        levels: std.ArrayList(usize),
        // Tombstones: tombstones.items[i] is true if node i is deleted.
        tombstones: std.ArrayList(bool),
        // Precomputed assignment probabilities for each layer.
        assignment_probabilities: std.ArrayList(f64),
        metric: *const fn (@Vector(N, T), @Vector(N, T)) f64,
        // Optional schema and metadata
        // The metadata for any vector begins at its id and ends at the lenght of the schema,
        // e.g., metadata[id..id + schema.len] is the metadata for the vector at id.
        // `Attrubute`s may not be nested. Moreover, attributes may only hold...
        // ...primitive types (booleans, integers, floats, strings, etc.)
        schema: ?std.ArrayList([]const u8),
        metadata: std.ArrayList(Metadata),
        allocator: std.mem.Allocator,

        pub fn init(
            l: ?usize,
            m: ?usize,
            ef_construction: ?usize,
            ef_search: ?usize,
            metric: ?Metric,
            schema: ?std.ArrayList([]const u8),
            allocator: std.mem.Allocator,
        ) !Self {
            const l_ = l orelse 5;
            const m_ = m orelse 16;
            const ef_construction_ = ef_construction orelse 200;
            const ef_search_ = ef_search orelse 50;
            const ml = 1.0 - (1.0 / @as(f64, @floatFromInt(m_)));
            // Initialize the assignProbas array with one entry per layer.
            var assignment_probabilities = std.ArrayList(f64).init(allocator);

            // For example, use an exponential decay: p(i) = exp(-i).
            for (0..l_) |i| {
                const f: f64 = @floatFromInt(i);
                try assignment_probabilities.append(@exp(-f));
            }

            const metric_ = if (metric) |f| switch (f) {
                Metric.L2 => &Self.euclideanDistance,
                Metric.Cosine => &Self.cosineSimilarity,
                Metric.InnerProduct => &Self.dotProduct,
            } else &Self.euclideanDistance;

            return Self{
                .layers = l_,
                .m = m_,
                .ml = ml,
                .ef_construction = ef_construction_,
                .ef_search = ef_search_,
                .vectors = std.ArrayList([N]T).init(allocator),
                .connections = std.ArrayList(ID).init(allocator),
                .offsets = std.ArrayList(usize).init(allocator),
                .levels = std.ArrayList(usize).init(allocator),
                .tombstones = std.ArrayList(bool).init(allocator),
                .assignment_probabilities = assignment_probabilities,
                .schema = schema,
                .metadata = std.ArrayList(Metadata).init(allocator),
                .metric = metric_,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.vectors.deinit();
            self.offsets.deinit();
            self.assignment_probabilities.deinit();
            self.levels.deinit();
            self.tombstones.deinit();
            self.connections.deinit();

            if (self.schema) |schema| {
                schema.deinit();
            }

            self.metadata.deinit();
        }

        pub fn search(self: *Self, query: [N]T, k: ?usize, filter: ?Filter) ![]Candidate {
            const k_ = k orelse 1;

            // If there are no vectors, return an empty list.
            if (self.vectors.items.len == 0) {
                var empty = std.ArrayList(Candidate).init(self.allocator);
                return empty.toOwnedSlice();
            }

            // Determine the top layer.
            var current_layer: usize = self.layers - 1;
            // Use a designated entry point.
            var entry: ID = 0;
            var candidate: Candidate = Candidate{ .distance = self.metric(query, self.vectors.items[entry]), .id = entry };

            // Descend from the top layer down to layer 1.
            while (current_layer > 0) {
                // Greedy search on the current layer with a small expansion factor (ef = 1)
                const layer_results = try self.knn(entry, query, 1, current_layer, filter);
                defer self.allocator.free(layer_results);
                if (layer_results.len == 0) break;
                // Take the best candidate from this layer.
                candidate = layer_results[0];
                // Skip if candidate is deleted.
                if (self.tombstones.items[candidate.id]) break;
                // Use it as the new entry point.
                entry = candidate.id;
                // Descend one layer.
                current_layer -= 1;
            }

            // At the bottom layer (layer 0), perform a full search using ef_search.
            const bottom_results = try self.knn(entry, query, self.ef_search, 0, filter);
            defer self.allocator.free(bottom_results);

            // Trim the result list to the top k_ candidates, ignoring deleted nodes.
            var trimmed = std.ArrayList(Candidate).init(self.allocator);
            errdefer trimmed.deinit();

            var count: usize = 0;
            for (bottom_results) |c| {
                if (!self.tombstones.items[c.id]) {
                    try trimmed.append(c);
                    count += 1;

                    if (count >= k_) {
                        break;
                    }
                }
            }

            return trimmed.toOwnedSlice();
        }

        fn knn(self: *Self, entry: ID, query: [N]T, ef: usize, layer: usize, filter: ?Filter) ![]Candidate {
            if (ef == 0) {
                return error.InvalidEF;
            }

            if (layer >= self.layers or layer < 0) {
                return error.InvalidLayer;
            }

            if (filter != null and self.schema == null) {
                return error.NoSchema;
            }

            if (filter) |f| {
                _ = try self.attributeIndex(f.attribute);
            }

            // Initialize the nearest-neighbors list and candidate heap as before.
            const best = Candidate{ .distance = self.metric(query, self.vectors.items[entry]), .id = entry };

            var nns = std.ArrayList(Candidate).init(self.allocator);
            errdefer nns.deinit();
            try nns.append(best);

            var visited = std.AutoHashMap(ID, struct {}).init(self.allocator);
            defer visited.deinit();
            try visited.put(entry, .{});

            var candidates = try MinMaxHeap(Candidate).init(self.allocator);
            defer candidates.deinit();
            try candidates.insert(best);

            while (candidates.len() > 0) {
                const candidate = candidates.deleteMin() catch break;

                // If candidate is worse than the worst in nns, we can stop.
                if (candidate.distance > nns.items[nns.items.len - 1].distance) break;

                // Compute neighbor range for candidate from offsets
                const start = self.offsets.items[candidate.id];
                const end = if (candidate.id + 1 < self.offsets.items.len)
                    self.offsets.items[candidate.id + 1]
                else
                    self.connections.items.len;

                // Iterate over candidate's neighbors.
                for (start..end) |i| {
                    const neighbor = self.connections.items[i];

                    // Only consider this neighbor if it exists in the current layer.
                    if (self.levels.items[neighbor] < layer) continue;

                    const distance = self.metric(query, self.vectors.items[neighbor]);
                    const current = Candidate{ .distance = distance, .id = neighbor };

                    // Skip the block if no attribute or filter
                    if (filter) |f| {
                        const attribute_index = try self.attributeIndex(f.attribute);
                        const metadata = self.metadata.items[neighbor + attribute_index];

                        if (!f.condition(metadata)) {
                            continue; // If it doesn't pass the filter, skip
                        }
                    }

                    if (!visited.contains(current.id)) {
                        try visited.put(neighbor, .{});

                        if (distance < nns.items[nns.items.len - 1].distance or nns.items.len < ef) {
                            try candidates.insert(current);
                            try insort(&nns, current);

                            if (nns.items.len > ef) {
                                _ = nns.pop();
                            }
                        }
                    }
                }
            }

            return nns.toOwnedSlice();
        }

        pub fn insert(self: *Self, vector: [N]T, metadata: ?std.ArrayList(Metadata), efc: ?usize, rng: *std.rand.DefaultPrng) !void {
            // Use the provided efc or the instance default.
            const efc_ = efc orelse self.ef_construction;

            // If the index is empty, insert the vector into all layers.
            if (self.vectors.items.len == 0) {
                // Append the vector and record that it exists on all layers.
                try self.vectors.append(vector);
                try self.levels.append(self.layers - 1);
                try self.tombstones.append(false);
                try self.offsets.append(0);

                if (metadata) |m| {
                    try self.metadata.appendSlice(m.items);
                }

                return;
            }

            // Determine new node’s level.
            const new_level = self.getInsertLayer(rng);
            // Append the new vector and record its level.
            try self.vectors.append(vector);
            const new_node_id = self.vectors.items.len - 1;
            try self.levels.append(new_level);
            try self.tombstones.append(false);

            if (metadata) |m| {
                try self.metadata.appendSlice(m.items);
            }

            // Starting point for search (entry) — here we use node 0 as the global entry.
            var entry: ID = 0;

            // Iterate over layers from bottom (layer 0) to top (layer self.layers-1).
            // (In many implementations, insertion is performed from the top down; adjust the loop
            // if you prefer that order.
            for (0..self.layers) |layer| {
                if (layer < new_level) {
                    // For layers below the new node’s level, do a greedy search with ef = 1.
                    const results = try self.knn(entry, vector, 1, layer, null);
                    defer self.allocator.free(results);

                    if (results.len > 0) {
                        entry = results[0].id;
                    }
                } else {
                    // For layers at or above newLevel, perform a full search with ef = efc_
                    const nns = try self.knn(entry, vector, efc_, layer, null);
                    defer self.allocator.free(nns);
                    // Create a new connection list for the new node in this layer.
                    var new_connections = std.ArrayList(ID).init(self.allocator);
                    defer new_connections.deinit();

                    for (nns) |candidate| {
                        try new_connections.append(candidate.id);
                        // Also update the candidate’s connections by adding the new node.
                        try self.addConnection(candidate.id, new_node_id);
                    }

                    // Insert the new node’s connection list for this layer.
                    try self.insertConnectionsForNode(new_node_id, new_connections);
                    // Optionally update the entry for the next layer to be the new node.
                    entry = new_node_id;
                }
            }
        }

        /// Builds the index from a list of vectors.
        pub fn create(
            self: *Self,
            vectors: std.ArrayList([N]T),
            metadata: ?std.ArrayList(std.ArrayList(Metadata)),
            efc: ?usize,
            rng: *std.rand.DefaultPrng,
        ) !void {
            if (vectors.items.len == 0) {
                return error.EmptyVectors;
            }

            if (metadata != null and self.schema != null) {
                if (metadata.?.items.len != vectors.items.len) {
                    return error.MetadataMismatch;
                }

                for (vectors.items, metadata.?.items) |vector, datum| {
                    try self.insert(vector, datum, efc, rng);
                }

                return;
            }

            for (vectors.items) |vector| {
                try self.insert(vector, null, efc, rng);
            }
        }

        /// Sample a level for a new node using the precomputed assignProbas.
        inline fn getInsertLayer(self: *Self, rng: *std.rand.DefaultPrng) usize {
            // First, compute the total probability.
            var total: f64 = 0;

            for (self.assignment_probabilities.items) |p| {
                total += p;
            }

            // Acquire a random stream from rng.
            const rand_stream = rng.random();

            // Generate a random value in [0, total)
            const r_val = rand_stream.float(f64) * total;
            var cumulative: f64 = 0;
            // Iterate over the assignProbas to choose a level.

            for (self.assignment_probabilities.items, 0..) |p, i| {
                cumulative += p;
                if (cumulative > r_val) {
                    return i;
                }
            }

            // Fallback: return the highest level.
            return self.assignment_probabilities.items.len - 1;
        }

        fn addConnection(self: *Self, node_id: ID, neighbor: ID) !void {
            if (node_id >= self.offsets.items.len) {
                return error.InvalidNodeID;
            }

            // Determine the end index of the neighbor list.
            // If there is a next node, use its offset; otherwise, use connections.items.len.
            const end = if (node_id + 1 < self.offsets.items.len)
                self.offsets.items[node_id + 1]
            else
                self.connections.items.len;

            // Insert the new neighbor at the end of nodeId's neighbor list.
            try self.connections.insert(end, neighbor);

            // Because we inserted an element, all offsets for nodes after nodeId must be increased by 1.
            for (node_id + 1..self.offsets.items.len) |i| {
                self.offsets.items[i] += 1;
            }
        }

        fn insertConnectionsForNode(self: *Self, node_id: ID, connections: std.ArrayList(ID)) !void {
            // In a well‐behaved insertion, the new node is always appended,
            // so its ID should equal self.vectors.items.len.
            if (node_id != self.vectors.items.len - 1) {
                return error.InvalidNodeID;
            }

            // Record the starting offset for this new node.
            const start = self.connections.items.len;
            try self.offsets.append(start);

            // Append each connection from connList into the flat connections array.
            for (connections.items) |connection| {
                try self.connections.append(connection);
            }
        }

        /// Finds the nearest neighbor to `vector` and deletes it (marks it as deleted).
        /// Tombstoned nodes are ignored during search.
        pub fn deleteNearest(self: *Self, vector: [N]T, filter: ?Filter) !void {
            const results = try self.search(vector, 1, filter);
            defer self.allocator.free(results);

            if (results.len > 0) {
                self.delete(results[0].id);
            }
        }

        /// Marks a node as deleted by setting its tombstone flag to true.
        fn delete(self: *Self, id: ID) void {
            self.tombstones.items[id] = true;
        }

        pub fn clean(self: *Self) !void {
            // Number of nodes in the current index.
            const old_count = self.vectors.items.len;

            // Allocate new array lists for the cleaned index.
            var new_vectors = std.ArrayList([N]T).init(self.allocator);
            var new_levels = std.ArrayList(usize).init(self.allocator);
            var new_tombstones = std.ArrayList(bool).init(self.allocator);
            var new_offsets = std.ArrayList(usize).init(self.allocator);
            var new_connections = std.ArrayList(ID).init(self.allocator);

            // We'll create a mapping from old node ID -> new node ID.
            // For deleted nodes, we set the mapping to std.math.maxInt(usize).
            var mapping = std.ArrayList(usize).init(self.allocator);
            defer mapping.deinit();

            var newId: usize = 0;
            var i: usize = 0;
            while (i < old_count) : (i += 1) {
                if (!self.tombstones.items[i]) {
                    // Node i is active; assign new ID.
                    try mapping.append(newId);
                    try new_vectors.append(self.vectors.items[i]);
                    try new_levels.append(self.levels.items[i]);
                    try new_tombstones.append(false);
                    newId += 1;
                } else {
                    // Mark deleted nodes with an invalid mapping.
                    try mapping.append(std.math.maxInt(usize));
                }
            }

            // Rebuild offsets and connections for active nodes.
            var current_offset: usize = 0;
            // Iterate over each old node. For each active node, re-map its neighbor list.
            i = 0;
            while (i < old_count) : (i += 1) {
                if (self.tombstones.items[i]) {
                    continue;
                }

                // For an active node i, record its new starting offset.
                try new_offsets.append(current_offset);

                // Determine neighbor range in the old flat connections array.
                // (Assuming that for node i, neighbors are stored from offsets[i] to
                //  (i+1 < offsets.len ? offsets[i+1] : connections.items.len))
                const start = self.offsets.items[i];
                const end = if (i + 1 < self.offsets.items.len)
                    self.offsets.items[i + 1]
                else
                    self.connections.items.len;

                var j: usize = start;
                while (j < end) : (j += 1) {
                    const oldNeighbor = self.connections.items[j];
                    // Only keep neighbors that are active.

                    if (oldNeighbor >= mapping.items.len) {
                        // Should not happen; skip for safety.
                        j += 1;
                        continue;
                    }

                    const new_neighbor = mapping.items[oldNeighbor];

                    if (new_neighbor == std.math.maxInt(usize)) {
                        // The neighbor was deleted; skip.
                        j += 1;
                        continue;
                    }

                    // Append the re-mapped neighbor ID.
                    try new_connections.append(new_neighbor);
                    current_offset += 1;
                }
            }

            // (Optional) If you want a final offset for the last node, it can be inferred
            // by newConnections.items.len. Many implementations use offsets as an array
            // of start indices only.

            // Replace the old arrays with the new ones.
            self.vectors = new_vectors;
            self.levels = new_levels;
            self.tombstones = new_tombstones;
            self.offsets = new_offsets;
            self.connections = new_connections;
        }

        fn attributeIndex(self: *Self, name: []const u8) !usize {
            if (self.schema == null) {
                return error.NoSchema;
            }

            // `.?` will always pass after the null check.
            if (self.schema.?.items.len == 0) {
                return error.EmptySchema;
            }

            for (self.schema.?.items, 0..) |attribute, i| {
                if (std.mem.eql(u8, attribute, name)) {
                    return i;
                }
            }

            return error.AttributeNotFound;
        }

        fn updateMetadata(self: *Self, id: ID, name: []const u8, value: Metadata) !void {
            const index = try self.attributeIndex(name);
            self.metadata.items[id + index] = value;
        }

        fn euclideanDistance(a: @Vector(N, T), b: @Vector(N, T)) f64 {
            var sum: f64 = 0.0;

            for (0..N) |i| {
                const diff = a[i] - b[i];
                sum += diff * diff;
            }

            return @sqrt(sum);
        }

        fn dotProduct(a: @Vector(N, T), b: @Vector(N, T)) f64 {
            var sum: f64 = 0.0;

            for (0..N) |i| {
                sum += a[i] * b[i];
            }

            return sum;
        }

        fn cosineSimilarity(a: @Vector(N, T), b: @Vector(N, T)) f64 {
            const dot = Self.dotProduct(a, b);
            const normA = Self.dotProduct(a, a);
            const normB = Self.dotProduct(b, b);
            return dot / (@sqrt(normA) * @sqrt(normB));
        }
    };
}

test "HNSW - basic initialization" {
    // We'll pick T=f64 and D=2 for a simple 2D example.
    const MyHNSW = HNSW(f64, 2);

    const allocator = std.heap.page_allocator;

    // Test the default init with all optional parameters = null
    var hnsw1 = try MyHNSW.init(null, null, null, null, null, null, allocator);
    defer hnsw1.deinit();

    try testing.expectEqual(5, hnsw1.layers);
    try testing.expectEqual(16, hnsw1.m);
    try testing.expectEqual(200, hnsw1.ef_construction);
    try testing.expectEqual(50, hnsw1.ef_search);

    // Test custom init
    var hnsw2 = try MyHNSW.init(3, 8, 100, 10, null, null, allocator);
    defer hnsw2.deinit();

    try testing.expectEqual(3, hnsw2.layers);
    try testing.expectEqual(8, hnsw2.m);
    try testing.expectEqual(100, hnsw2.ef_construction);
    try testing.expectEqual(10, hnsw2.ef_search);
}

test "HNSW - insert and search" {
    const MyHNSW = HNSW(f64, 2);

    const allocator = std.heap.page_allocator;

    var hnsw = try MyHNSW.init(null, null, null, null, null, null, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(42);

    // Insert a few 2D points
    // e.g., points forming a "square" around (0,0) and one in the center.
    const points = [_][2]f64{
        .{ -1.0, -1.0 }, // p0
        .{ -1.0, 1.0 }, // p1
        .{ 1.0, 1.0 }, // p2
        .{ 1.0, -1.0 }, // p3
        .{ 0.0, 0.0 }, // p4 (center)
    };

    for (points) |pt| {
        try hnsw.insert(pt, null, null, &rng);
    }

    // Verify we have 5 vectors stored
    try testing.expectEqual(@as(usize, 5), hnsw.vectors.items.len);

    // Search for neighbors of the point near (0.0, 0.0)
    {
        const query = .{ 0.1, 0.0 }; // Slightly offset from center
        // Search for 2 nearest neighbors
        const results = try hnsw.search(query, 2, null);
        defer allocator.free(results);

        // Expect at least 1 result
        try testing.expect(results.len >= 1);

        // The top result is probably the center (0.0, 0.0)
        // Or something extremely close
        const best_candidate = results[0];
        // Because we appended in order, ID 4 is (0.0, 0.0)
        try testing.expectEqual(4, best_candidate.id);
    }
}

test "HNSW - delete nearest and verify tombstone" {
    const MyHNSW = HNSW(f64, 2);

    const allocator = std.heap.page_allocator;

    var hnsw = try MyHNSW.init(null, null, null, null, null, null, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(1234);

    // Insert points
    const points = [_][2]f64{
        .{ 0.0, 0.0 }, // ID=0
        .{ 1.0, 1.0 }, // ID=1
        .{ -1.0, -1.0 }, // ID=2
    };

    for (points) |pt| {
        try hnsw.insert(pt, null, null, &rng);
    }

    try testing.expectEqual(3, hnsw.vectors.items.len);

    // Before deletion, searching near (1,1) should return ID=1 as nearest
    {
        const query = .{ 1.0, 1.1 };
        const results = try hnsw.search(query, 1, null);
        defer allocator.free(results);
        try testing.expectEqual(1, results[0].id);
    }

    // Now delete nearest to (1,1) => that should tombstone ID=1
    try hnsw.deleteNearest(.{ 1.0, 1.0 }, null);

    // Searching near (1,1) now should return something else, e.g. ID=0 or ID=2
    // but definitely not ID=1
    {
        const query = .{ 1.0, 1.0 };
        const results = try hnsw.search(query, 1, null);
        defer allocator.free(results);

        try testing.expect(results.len >= 1);
        try testing.expect(results[0].id != 1); // ID=1 is tombstoned
    }
}

test "HNSW - clean tombstoned nodes" {
    const MyHNSW = HNSW(f64, 2);

    const allocator = std.heap.page_allocator;

    var hnsw = try MyHNSW.init(null, null, null, null, null, null, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(9876);

    // Insert some points
    const points = [_][2]f64{
        .{ 2.0, 2.0 }, // ID=0
        .{ -2.0, 2.0 }, // ID=1
        .{ -2.0, -2.0 }, // ID=2
        .{ 2.0, -2.0 }, // ID=3
    };

    for (points) |pt| {
        try hnsw.insert(pt, null, null, &rng);
    }
    try testing.expectEqual(4, hnsw.vectors.items.len);

    // Delete two points
    hnsw.delete(1); // tombstone ID=1
    hnsw.delete(2); // tombstone ID=2

    // Check that tombstones are set
    try testing.expect(hnsw.tombstones.items[1]);
    try testing.expect(hnsw.tombstones.items[2]);

    // Clean up. Should remove tombstoned nodes and re-map IDs.
    try hnsw.clean();

    // Now we should have only 2 active vectors (IDs get remapped).
    try testing.expectEqual(@as(usize, 2), hnsw.vectors.items.len);

    // They should correspond to the old IDs=0 and old ID=3,
    // but we don't necessarily know which new ID is which.
    // The important thing: tombstoned = false for all that remain
    for (hnsw.tombstones.items) |flag| {
        try testing.expect(!flag);
    }
}

test "HNSW - distance metrics (euclidean, cosine similarity, dot product)" {
    const MyHNSW = HNSW(f64, 2);
    const allocator = std.heap.page_allocator;

    // Initialize HNSW
    var hnsw = try MyHNSW.init(null, null, null, null, null, null, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(1234);

    // Insert 3 points forming an equilateral triangle with side length ~2, centered at origin.
    // Each point is 120° from the others.
    // - Euclidean distance between any two: ~2.0
    // - Cosine similarity between any two: cos(120°) = -0.5
    // - Dot product between any two: |a||b|cos(120°) = 1*1*(-0.5) = -0.5.
    //   (Here, each point has norm=1.0, so the dot is simply -0.5.)
    const points = [_][2]f64{
        .{ 1.0, 0.0 }, // ID=0
        .{ -0.5, 0.866 }, // ID=1
        .{ -0.5, -0.866 }, // ID=2
    };

    for (points) |pt| {
        try hnsw.insert(pt, null, null, &rng);
    }

    try std.testing.expectEqual(@as(usize, 3), hnsw.vectors.items.len);

    // --------------------- Euclidean Distance ---------------------
    {
        // Between distinct points => ~1.732 (since side ~2)
        for (0..3) |i| {
            for (i + 1..3) |j| {
                const dist = MyHNSW.euclideanDistance(hnsw.vectors.items[i], hnsw.vectors.items[j]);
                try std.testing.expectApproxEqRel(1.73203810582, dist, 1e-3);
            }
        }
        // Distance to itself => 0.0
        for (0..3) |i| {
            const dist = MyHNSW.euclideanDistance(hnsw.vectors.items[i], hnsw.vectors.items[i]);
            try std.testing.expectApproxEqRel(0.0, dist, 1e-3);
        }
    }

    // --------------------- Cosine Similarity ----------------------
    {
        // Between distinct points => -0.5 (angle=120°, cos(120°) = -0.5)
        for (0..3) |i| {
            for (i + 1..3) |j| {
                const sim = MyHNSW.cosineSimilarity(hnsw.vectors.items[i], hnsw.vectors.items[j]);
                try std.testing.expectApproxEqRel(-0.5, sim, 1e-3);
            }
        }
        // With itself => 1.0
        for (0..3) |i| {
            const sim = MyHNSW.cosineSimilarity(hnsw.vectors.items[i], hnsw.vectors.items[i]);
            try std.testing.expectApproxEqRel(1.0, sim, 1e-3);
        }
    }

    // --------------------- Dot Product ----------------------------
    {
        // Between distinct points => -0.5
        for (0..3) |i| {
            for (i + 1..3) |j| {
                const dot = MyHNSW.dotProduct(hnsw.vectors.items[i], hnsw.vectors.items[j]);
                try std.testing.expectApproxEqRel(-0.5, dot, 1e-3);
            }
        }
        // With itself => 1.0 (since each vector's length is 1.0)
        for (0..3) |i| {
            const dot = MyHNSW.dotProduct(hnsw.vectors.items[i], hnsw.vectors.items[i]);
            try std.testing.expectApproxEqRel(1.0, dot, 1e-3);
        }
    }
}

// Example of a simple filter that checks a single attribute for a string match.
// Adjust to your own Filter type if it differs.
fn colorFilter(color: []const u8) Filter {
    return Filter{
        .attribute = "color", // the attribute name we want to match
        .condition = struct {
            fn call(value: Metadata) bool {
                // We interpret the value as a string. If it's not `.str`, return false.
                switch (value) {
                    .str => return std.mem.eql(u8, value.str, color),
                    else => return false,
                }
            }
        }.call,
    };
}

test "metadata - basic insert and filter" {
    const allocator = std.testing.allocator;

    // We pretend our schema is simply one attribute named `color`.
    // If your real schema is an array of attribute descriptors,
    // replace with something that matches your code’s representation.
    // For example, we might store it as a single string "color".
    var schema = std.ArrayList([]const u8).init(allocator);
    try schema.append("color");

    // Create an HNSW index for 2D vectors (f32). Adjust dimension & type as needed.
    var hnsw = try HNSW(f32, 2).init(
        null, // layers (use default)
        null, // m (use default)
        null, // ef_construction (use default)
        null, // ef_search (use default)
        null,
        schema, // our schema
        allocator,
    );
    defer hnsw.deinit();

    // Insert a couple of 2D vectors with different "color" metadata
    // For illustration, we assume you have a function to store metadata in `hnsw.metadata`.
    // We also assume you have a PRNG for insertion if needed. For a deterministic test, you can use a fixed seed.
    var rng = std.rand.DefaultPrng.init(1234);

    // Insert first vector: [1.0, 2.0], color= "blue"
    var meta_a = std.ArrayList(Metadata).init(allocator);
    defer meta_a.deinit();
    try meta_a.append(.{ .str = "blue" });
    try hnsw.insert(.{ 1.0, 2.0 }, meta_a, null, &rng);
    // Suppose the new vector got ID = 0. Now store the metadata:

    // Insert second vector: [2.0, 3.0], color= "red"
    var meta_b = std.ArrayList(Metadata).init(allocator);
    defer meta_b.deinit();
    try meta_b.append(.{ .str = "red" });
    try hnsw.insert(.{ 2.0, 3.0 }, meta_b, null, &rng);
    // Suppose the new vector got ID = 1.

    // Insert third vector: [10.0, 10.0], color= "blue"
    var meta_c = std.ArrayList(Metadata).init(allocator);
    defer meta_c.deinit();
    try meta_c.append(.{ .str = "blue" });
    try hnsw.insert(.{ 10.0, 10.0 }, meta_c, null, &rng);
    // Suppose the new vector got ID = 2.

    // Create a filter that matches color = "blue"
    const blue_filter = colorFilter("blue");

    // Now search for neighbors near [1.5, 2.5], but only with color="blue".
    // We set k=10 for demonstration.
    const results = try hnsw.search(.{ 1.5, 2.5 }, 10, blue_filter);
    defer allocator.free(results);

    // We expect vectors ID=0 and ID=2 to pass the color="blue" filter.
    // Of those, the vector at (1,2) is closer to (1.5, 2.5) than the one at (10,10).
    // The result set might contain both, but sorted by distance typically.
    // Let's just ensure we find ID=0 somewhere, and that ID=1 is not in the results.
    var seen_id_0 = false;
    var seen_id_1 = false;
    var seen_id_2 = false;

    for (results) |cand| {
        if (cand.id == 0) seen_id_0 = true;
        if (cand.id == 1) seen_id_1 = true;
        if (cand.id == 2) seen_id_2 = true;
    }

    try testing.expect(seen_id_0);
    try testing.expect(!seen_id_1); // "red" vector should be filtered out
    try testing.expect(seen_id_2);
}

// Demonstrates what happens if we pass a filter with no schema set (we expect an error).
test "metadata - no schema, should fail if filter is used" {
    const allocator = std.testing.allocator;

    // We pass `null` for the schema.
    var hnsw = try HNSW(f32, 2).init(null, null, null, null, null, null, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(5678);
    try hnsw.insert(.{ 0.0, 0.0 }, null, null, &rng);

    // Create a filter referencing "color" again.
    const blue_filter = colorFilter("blue");

    // Attempting to search with a filter while schema is null => expect error.NoSchema.
    try testing.expectError(error.NoSchema, hnsw.search(.{ 0.0, 0.0 }, 1, blue_filter));
}

// Demonstrate attempting to filter on an attribute that doesn't exist in the schema.
test "metadata - attribute not found" {
    const allocator = std.testing.allocator;

    // Suppose our schema is just "color" again:
    var schema = std.ArrayList([]const u8).init(allocator);
    try schema.append("color");

    var hnsw = try HNSW(f32, 2).init(null, null, null, null, null, schema, allocator);
    defer hnsw.deinit();

    var rng = std.rand.DefaultPrng.init(555);
    var metadata = std.ArrayList(Metadata).init(allocator);
    defer metadata.deinit();
    try metadata.append(.{ .str = "blue" });
    try hnsw.insert(.{ 5.0, 5.0 }, metadata, null, &rng);

    // Now define a filter for a nonexistent attribute "category".
    const filter = Filter{
        .attribute = "category",
        .condition = struct {
            fn call(value: Metadata) bool {
                // arbitrary logic
                _ = value;
                return true;
            }
        }.call,
    };

    try testing.expectError(error.AttributeNotFound, hnsw.search(.{ 5.0, 5.0 }, 1, filter));
}
