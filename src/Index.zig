const std = @import("std");
const errors = @import("errors.zig");

const TableEntry = struct {
    key: []const u8,
    value: type,
};

const Metric = enum {
    euclidean,
    cosine,
    dot_product,
};

/// Represents a vector index.
///
/// The user must specify the dimension of the vectors and
/// supply a collection of string-type pairs. The collection
/// should be an iterable of tuples where the first element
/// is a `[]const u8` (string) and the second element is any
/// type `T`.
///
/// For example, the collection can be a `[]const ([]const u8, T)`,
/// or any other iterable where each element is a tuple
/// containing `[]const u8` and a corresponding type `T`.
pub fn Index() !type {
    return struct {
        const Self = @This();

        fn init(dim: u64, table: anytype, metric: Metric) !Self {
            if (dim == 0) {
                return errors.DimensionZero;
            }

            const calc_dim = dim + table.size;

            const index = Self{
                .dim = calc_dim,
                .table = table,
                .metric = metric,
            };

            index.pack(&table);

            return index;
        }

        pub fn initFromVector() !Self {}

        /// The user may supply us a table with all
        /// manner of types. Our job is to order the
        /// table in a way that is most efficient in
        /// terms of memory footprint and speed.
        /// Moreover, these attributes will be
        /// stored directly within each vector;
        /// they need to be ordered in the most
        /// compact way possible. Remember, accesses
        /// are done 8 bytes (64 bits) at a time.
        /// So, if we can order the table in a way
        /// that minimizes cache misses, we can
        /// speed up our search.
        ///
        /// The algorithm is as follows:
        /// 1. Sort first by size // 8
        /// 2. Sort second by size % 8
        /// Yielding the following order:
        /// [8, 4, 1, 11] -> // -> [1, 1, 0, 0]
        /// [8, 4, 1, 11] -> % -> [0, 3, 1, 4]
        /// Resulting in:
        /// | 8, 1, 0|
        /// |11, 1, 3|
        /// | 1, 0, 1|
        /// | 4, 0, 4|
        fn pack(self: *Self) !void {
            const n = self.len;
            var matrix: [3][n]u32 = undefined;

            // Get sizes
            var sizes: [n]u32 = undefined;
            for (self, 0..n) |entry, i| {
                sizes[i] = @sizeOf(entry.value);
            }
            matrix[0] = sizes;

            // Sort on size / 8
            var div: [n]u32 = undefined;
            for (sizes, 0..n) |size, i| {
                div[i] = @divTrunc(size, 8);
            }
            std.mem.sort(u32, &div, comptime std.sort.desc(u32));
            matrix[1] = div;

            // Sort on size % 8
            var mod: [n]u32 = undefined;
            for (sizes, 0..n) |size, i| {
                mod[i] = @mod(size, 8);
            }
            std.mem.sort(u32, &mod, comptime std.sort.desc(u32));
            matrix[2] = mod;
        }
    };
}

test "index creation and packing" {
    var table: [3]TableEntry = [_]TableEntry{
        .{ .key = "A", .element = "Element1", .size = 0 },
        .{ .key = "B", .element = "Element2", .size = 0 },
        .{ .key = "C", .element = "Element3", .size = 0 },
    };

    var index = try Index.init(10, table[0..]);
    index.pack();
}
