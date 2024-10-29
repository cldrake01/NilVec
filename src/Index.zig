const std = @import("std");

/// Represents an HNSW vector index.
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
const Index = struct {
    dim: u64,
    table: []TableEntry,
    metric: ?Metric = null,

    pub fn init(dim: u64, table: []TableEntry) !Index {
        if (dim == 0) {
            return error.DimensionZero;
        }

        const calc_dim = dim + table.len;
        return Index{
            .dim = calc_dim,
            .table = table,
            .metric = null,
        };
    }

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
    pub fn pack(self: *Index) void {
        // Create a copy of the table to pack it
        var rearranged = std.heap.auto_vec([]TableEntry, std.heap.page_allocator);
        defer rearranged.deinit();

        // Load elements into packed table
        for (self.table) |entry| {
            _ = rearranged.append(TableEntry{ .key = entry.key, .size = @sizeOf(entry.element) - 8 });
        }

        // Sort by size in ascending order
        std.sort.sort(rearranged.items, TableEntry.compare);
    }
};

const TableEntry = struct {
    key: []const u8,
    element: []const u8,
    size: usize,

    pub fn compare(a: *TableEntry, b: *TableEntry) i32 {
        const a_size: i32 = @intCast(a.size);
        const b_size: i32 = @intCast(b.size);

        return a_size - b_size;
    }
};

const Metric = struct {
    // Define fields based on your requirements
};

test "index creation and packing" {
    var table: [3]TableEntry = [_]TableEntry{
        .{ .key = "A", .element = "Element1", .size = 0 },
        .{ .key = "B", .element = "Element2", .size = 0 },
        .{ .key = "C", .element = "Element3", .size = 0 },
    };

    var index = try Index.init(10, table[0..]);
    index.pack();
}
