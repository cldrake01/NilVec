const std = @import("std");
const testing = std.testing;

const MinMaxHeap = @import("MinMaxHeap.zig").MinMaxHeap;
const Metadata = @import("Metadata.zig").Metadata;
const Candidate = @import("Candidate.zig").Candidate;
const Metric = @import("Metric.zig").Metric;
const ID = @import("ID.zig").ID;
const Filter = @import("Filter.zig").Filter;

pub fn Flat(comptime T: type, comptime D: usize) type {
    comptime {
        switch (@typeInfo(T)) {
            .Int, .Float, .ComptimeInt, .ComptimeFloat => {},
            else => {
                @compileError("HNSW only supports integer and floating point types.");
            },
        }

        if (D == 0) {
            @compileError("Dimension cannot be zero.");
        }
    }

    return struct {
        const Self = @This();

        vectors: std.ArrayList([D]T),
        // Tombstones: tombstones.items[i] is true if node i is deleted.
        tombstones: std.ArrayList(bool),
        metric: *const fn (@Vector(D, T), @Vector(D, T)) f64,
        // Optional schema and metadata
        // The metadata for any vector begins at its id and ends at the lenght of the schema,
        // e.g., metadata[id..id + schema.len] is the metadata for the vector at id.
        // `Attrubute`s may not be nested. Moreover, attributes may only hold...
        // ...primitive types (booleans, integers, floats, strings, etc.)
        schema: ?std.ArrayList([]const u8),
        metadata: std.ArrayList(Metadata),
        allocator: std.mem.Allocator,

        pub fn init(metric: ?Metric, schema: ?std.ArrayList([]const u8), allocator: std.mem.Allocator) Self {
            const metric_ = if (metric) |f| switch (f) {
                Metric.L2 => &Self.euclideanDistance,
                Metric.Cosine => &Self.cosineSimilarity,
                Metric.InnerProduct => &Self.dotProduct,
            } else &Self.euclideanDistance;

            return Self{
                .vectors = std.ArrayList([D]T).init(allocator),
                .tombstones = std.ArrayList(bool).init(allocator),
                .metric = metric_,
                .schema = schema,
                .metadata = std.ArrayList(Metadata).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.vectors.deinit();
            self.tombstones.deinit();

            if (self.schema) |schema| {
                schema.deinit();
            }

            self.metadata.deinit();
        }

        pub fn search(self: *Self, query: [D]T, k: ?usize, filter: ?Filter) ![]Candidate {
            const k_ = k orelse 1;
            var heap = try MinMaxHeap(Candidate).init(self.allocator);
            defer heap.deinit();

            if (self.vectors.items.len == 0) {
                return error.EmptyIndex;
            }

            for (self.vectors.items, 0..) |vector, i| {
                const candidate = Candidate{ .distance = self.metric(query, vector), .id = i };
                try heap.insert(candidate);
            }

            var count: usize = 0;
            var results = std.ArrayList(Candidate).init(self.allocator);
            defer results.deinit();

            while (heap.len() > 0) {
                const candidate = try heap.deleteMin();

                // Skip the block if no attribute or filter
                if (filter) |f| {
                    const attribute_index = try self.attributeIndex(f.attribute);
                    const metadata = self.metadata.items[candidate.id + attribute_index];

                    if (!f.condition(metadata)) {
                        continue; // If it doesn't pass the filter, skip
                    }
                }

                if (!self.tombstones.items[candidate.id]) {
                    try results.append(candidate);
                    count += 1;

                    if (count >= k_) {
                        break;
                    }
                }
            }

            return results.toOwnedSlice();
        }

        pub fn insert(self: *Self, vector: [D]T, metadata: ?std.ArrayList(Metadata)) !void {
            try self.vectors.append(vector);
            try self.tombstones.append(false);

            if (metadata) |m| {
                try self.metadata.appendSlice(m.items);
            }
        }

        /// Builds the index from a set of vectors.
        pub fn create(self: *Self, vectors: std.ArrayList([D]T), metadata: ?std.ArrayList(std.ArrayList(Metadata))) !void {
            if (vectors.items.len == 0) {
                return error.EmptyVectors;
            }

            if (metadata != null and self.schema != null) {
                if (metadata.?.items.len != vectors.items.len) {
                    return error.MetadataMismatch;
                }

                for (vectors.items, metadata.?.items) |vector, datum| {
                    try self.insert(vector, datum);
                }

                return;
            }

            for (vectors.items) |vector| {
                try self.insert(vector, null);
            }
        }

        /// Marks the nearest node as deleted.
        pub fn deleteNearest(self: *Self, query: [D]T) !void {
            const nearest = try self.search(query, 1, null);
            defer self.allocator.free(nearest);

            if (nearest.len > 0) {
                self.delete(nearest[0].id);
            }
        }

        /// Marks a node as deleted by setting its tombstone flag to true.
        fn delete(self: *Self, id: ID) void {
            self.tombstones.items[id] = true;
        }

        /// Cleans the index by removing all tombstoned nodes.
        pub fn clean(self: *Self) !void {
            var write_index: usize = 0;
            // Determine the number of metadata entries per vector, if any.
            var meta_count: usize = 0;

            if (self.schema) |schema| {
                meta_count = schema.items.len;
            }

            // Iterate over all vectors.
            for (self.vectors.items, 0..) |vector, i| {
                if (!self.tombstones.items[i]) {

                    // Only perform a move if necessary.
                    if (write_index != i) {
                        self.vectors.items[write_index] = vector;
                        self.tombstones.items[write_index] = false; // Reset tombstone flag.
                        // If metadata is enabled, copy the associated block.

                        if (self.schema) |_| {
                            const src_start = i * meta_count;
                            const dst_start = write_index * meta_count;
                            @memcpy(
                                self.metadata.items[dst_start .. dst_start + meta_count],
                                self.metadata.items[src_start .. src_start + meta_count],
                            );
                        }
                    }

                    write_index += 1;
                }
            }

            // Reslice the arrays to remove the deleted items.
            self.vectors.items = self.vectors.items[0..write_index];
            self.tombstones.items = self.tombstones.items[0..write_index];

            if (self.schema) |_| {
                self.metadata.items = self.metadata.items[0 .. write_index * meta_count];
            }
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

        fn euclideanDistance(a: @Vector(D, T), b: @Vector(D, T)) f64 {
            var sum: f64 = 0.0;

            for (0..D) |i| {
                const diff = a[i] - b[i];
                sum += diff * diff;
            }

            return @sqrt(sum);
        }

        fn dotProduct(a: @Vector(D, T), b: @Vector(D, T)) f64 {
            var sum: f64 = 0.0;

            for (0..D) |i| {
                sum += a[i] * b[i];
            }

            return sum;
        }

        fn cosineSimilarity(a: @Vector(D, T), b: @Vector(D, T)) f64 {
            const dot = Self.dotProduct(a, b);
            const normA = Self.dotProduct(a, a);
            const normB = Self.dotProduct(b, b);
            return dot / (@sqrt(normA) * @sqrt(normB));
        }
    };
}

/// Example "colorFilter" helper, matching a `Metadata.str` field against a target color string.
fn colorFilter(color: []const u8) Filter {
    return Filter{
        .attribute = "color",
        .condition = struct {
            fn call(value: Metadata) bool {
                switch (value) {
                    .str => return std.mem.eql(u8, value.str, color),
                    else => return false,
                }
            }
        }.call,
    };
}

test "Flat - basic initialization" {
    // We'll pick T = f64 and dimension = 2 for a simple 2D index.
    const MyFlat = Flat(f64, 2);

    const allocator = std.testing.allocator;
    var index = MyFlat.init(null, null, allocator);
    defer index.deinit();

    // Expect that internal arrays are empty upon initialization
    try testing.expectEqual(0, index.vectors.items.len);
    try testing.expectEqual(0, index.tombstones.items.len);
}

test "Flat - insert and search" {
    const MyFlat = Flat(f64, 2);
    const allocator = std.testing.allocator;

    var index = MyFlat.init(null, null, allocator);
    defer index.deinit();

    // Insert a few 2D points
    const points = [_][2]f64{
        .{ -1.0, -1.0 }, // 0
        .{ -1.0, 1.0 }, // 1
        .{ 1.0, 1.0 }, // 2
        .{ 1.0, -1.0 }, // 3
        .{ 0.0, 0.0 }, // 4
    };

    // Insert them without metadata
    for (points) |pt| {
        try index.insert(pt, null);
    }
    try testing.expectEqual(5, index.vectors.items.len);

    // Search for k=2 neighbors near (0.1, 0.1).
    const query = .{ 0.1, 0.1 };
    const results = try index.search(query, 2, null);
    defer allocator.free(results);

    // We should find at least 1 result. The nearest likely is ID=4 (the center).
    try testing.expect(results.len >= 1);
    try testing.expectEqual(4, results[0].id);
}

test "Flat - delete nearest and verify tombstone" {
    const MyFlat = Flat(f64, 2);
    const allocator = std.testing.allocator;

    var index = MyFlat.init(null, null, allocator);
    defer index.deinit();

    // Insert a few points
    const points = [_][2]f64{
        .{ 0.0, 0.0 }, // ID=0
        .{ 1.0, 1.0 }, // ID=1
        .{ -1.0, -1.0 }, // ID=2
    };

    for (points) |pt| {
        try index.insert(pt, null);
    }
    try testing.expectEqual(3, index.vectors.items.len);

    // Confirm nearest to (1, 1) is ID=1
    {
        const query = .{ 1.0, 1.0 };
        const results = try index.search(query, 1, null);
        defer allocator.free(results);

        try testing.expectEqual(1, results[0].id);
    }

    // Now deleteNearest for (1,1). That should tombstone ID=1.
    try index.deleteNearest(.{ 1.0, 1.0 });

    // Confirm tombstone for ID=1 is set
    try testing.expect(index.tombstones.items[1]);

    // Searching near (1,1) should no longer return ID=1
    {
        const query = .{ 1.0, 1.0 };
        const results = try index.search(query, 1, null);
        defer allocator.free(results);

        try testing.expect(results.len >= 1);
        try testing.expect(results[0].id != 1);
    }
}

test "Flat - clean tombstoned nodes" {
    const MyFlat = Flat(f64, 2);
    const allocator = std.testing.allocator;

    var index = MyFlat.init(null, null, allocator);
    defer index.deinit();

    // Insert 4 points
    const points = [_][2]f64{
        .{ 2.0, 2.0 }, // ID=0
        .{ -2.0, 2.0 }, // ID=1
        .{ -2.0, -2.0 }, // ID=2
        .{ 2.0, -2.0 }, // ID=3
    };
    for (points) |pt| {
        try index.insert(pt, null);
    }

    // Tombstone 2 of them
    index.tombstones.items[1] = true; // ID=1
    index.tombstones.items[2] = true; // ID=2

    // Now clean
    try index.clean();

    // Only 2 remain
    try testing.expectEqual(2, index.vectors.items.len);
    // All surviving tombstone flags should be false
    for (index.tombstones.items) |flag| {
        try testing.expect(!flag);
    }
}

test "Flat - distance functions (euclidean, dotProduct, cosineSimilarity)" {
    const MyFlat = Flat(f64, 2);
    // We only want the static methods, so we don't even need to allocate an index here.
    // But let's do it for consistency.
    const allocator = std.testing.allocator;
    var index = MyFlat.init(null, null, allocator);
    defer index.deinit();

    // We'll define three 2D points forming an equilateral triangle of side 2,
    // each 120° from the others. Distance between any pair is ~2.0
    const a = [2]f64{ 1.0, 0.0 };
    const b = [2]f64{ -0.5, 0.866 };
    const c = [2]f64{ -0.5, -0.866 };

    // --------------------- Euclidean Distance ---------------------
    {
        const d_ab = MyFlat.euclideanDistance(a, b);
        const d_ac = MyFlat.euclideanDistance(a, c);
        const d_bc = MyFlat.euclideanDistance(b, c);

        // side length ~ 2.0, but let's allow for rounding
        try testing.expectApproxEqRel(1.7320381058163818e0, d_ab, 1e-3);
        try testing.expectApproxEqRel(1.7320381058163818e0, d_ac, 1e-3);
        try testing.expectApproxEqRel(1.7320381058163818e0, d_bc, 1e-3);

        // distance to self => 0
        try testing.expectApproxEqRel(0.0, MyFlat.euclideanDistance(a, a), 1e-12);
    }

    // --------------------- Dot Product ----------------------------
    // For these points, the angle between them is 120°, so dot(a,b) = |a||b|cos(120°) => 1*1*(-0.5) = -0.5
    {
        const dot_ab = MyFlat.dotProduct(a, b);
        try testing.expectApproxEqRel(-0.5, dot_ab, 1e-3);

        // dot(a,a) => 1.0
        try testing.expectApproxEqRel(1.0, MyFlat.dotProduct(a, a), 1e-12);
    }

    // --------------------- Cosine Similarity ----------------------
    // angle=120° => cos(120°) = -0.5
    {
        const sim_ab = MyFlat.cosineSimilarity(a, b);
        try testing.expectApproxEqRel(-0.5, sim_ab, 1e-3);

        // similarity with self => 1.0
        try testing.expectApproxEqRel(1.0, MyFlat.cosineSimilarity(a, a), 1e-12);
    }
}

test "Flat - metadata and simple filter" {
    const MyFlat = Flat(f64, 2);
    const allocator = std.testing.allocator;

    // Create a minimal schema with one attribute: "color".
    var schema = std.ArrayList([]const u8).init(allocator);
    try schema.append("color");

    // Initialize the index with that schema
    var index = MyFlat.init(Metric.Cosine, schema, allocator);
    defer index.deinit();

    // Insert two vectors with different color attributes
    {
        var meta_blue = std.ArrayList(Metadata).init(allocator);
        defer meta_blue.deinit();
        try meta_blue.append(.{ .str = "blue" });
        try index.insert(.{ 0.0, 0.0 }, meta_blue);
    }
    {
        var meta_red = std.ArrayList(Metadata).init(allocator);
        defer meta_red.deinit();
        try meta_red.append(.{ .str = "red" });
        try index.insert(.{ 10.0, 10.0 }, meta_red);
    }

    // Filter on color="blue"
    const blue_filter = colorFilter("blue");

    // Searching near (0.1,0.0) with the filter => expect we find only the blue vector (ID=0)
    const results = try index.search(.{ 0.1, 0.0 }, 2, blue_filter);
    defer allocator.free(results);

    // We expect ID=0 in results, but not ID=1
    var found_blue = false;
    var found_red = false;
    for (results) |candidate| {
        if (candidate.id == 0) found_blue = true;
        if (candidate.id == 1) found_red = true;
    }
    try testing.expect(found_blue);
    try testing.expect(!found_red);
}

test "Flat - no schema, expect error if filter is used" {
    const MyFlat = Flat(f32, 2);
    const allocator = std.testing.allocator;

    // Pass null for schema
    var index = MyFlat.init(Metric.L2, null, allocator);
    defer index.deinit();

    // Insert a single vector with no metadata
    try index.insert(.{ 1.0, 2.0 }, null);

    // Attempt to search with a filter => expect error.NoSchema
    const blue_filter = colorFilter("blue");
    try testing.expectError(error.NoSchema, index.search(.{ 1.0, 2.0 }, 1, blue_filter));
}

test "Flat - attribute not found" {
    const MyFlat = Flat(f32, 2);
    const allocator = std.testing.allocator;

    // Suppose our schema only has "color".
    var schema = std.ArrayList([]const u8).init(allocator);
    try schema.append("color");

    var index = MyFlat.init(Metric.L2, schema, allocator);
    defer index.deinit();

    // Insert [5.0,5.0] with color="blue"
    {
        var meta = std.ArrayList(Metadata).init(allocator);
        defer meta.deinit();
        try meta.append(.{ .str = "blue" });
        try index.insert(.{ 5.0, 5.0 }, meta);
    }

    // Define a filter for a missing attribute "category"
    const missing_attr_filter = Filter{
        .attribute = "category",
        .condition = struct {
            fn call(value: Metadata) bool {
                _ = value;
                return true;
            }
        }.call,
    };

    try testing.expectError(error.AttributeNotFound, index.search(.{ 5.0, 5.0 }, 1, missing_attr_filter));
}
