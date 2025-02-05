const std = @import("std");

/// Error enumeration for our heap.
pub const Error = error{EmptyHeap};

/// Generic MinMaxHeap that supports numeric types and structs with an `order()` method.
/// If T is a struct, it must have a method:
///     pub fn order(self: T) f64 { ... }
pub fn MinMaxHeap(comptime T: type) type {
    comptime {
        // Allow numeric types or structs.
        switch (@typeInfo(T)) {
            .Int, .Float, .ComptimeInt, .ComptimeFloat, .Struct => {},
            else => @compileError("MinMaxHeap only supports numeric types or structs with an `order()` method."),
        }
    }

    return struct {
        const Self = @This();

        allocator: *const std.mem.Allocator,
        data: std.ArrayList(T),

        /// Initialize a new heap.
        pub fn init(allocator: std.mem.Allocator) !Self {
            return Self{
                .allocator = &allocator,
                .data = std.ArrayList(T).init(allocator),
            };
        }

        /// Free resources used by the heap.
        pub fn deinit(self: *Self) void {
            self.data.deinit();
        }

        /// Returns the current length of the heap.
        pub fn len(self: *Self) usize {
            return self.data.items.len;
        }

        /// Returns the ordering key for a given value.
        fn key(value: T) f64 {
            // If T is a struct, we expect an 'order()' method.
            if (@typeInfo(T) == .Struct) {
                return T.order(value);
            }
            // For numeric types, convert value to f64.
            return @as(f64, value);
        }

        /// Compute the level of a node (0-based). Level is floor(log₂(index+1)).
        fn level(index: usize) usize {
            var lvl: usize = 0;
            var i = index + 1;
            while (i > 1) {
                i /= 2;
                lvl += 1;
            }
            return lvl;
        }

        /// Return the index of the parent node or null if at the root.
        pub fn parentIndex(index: usize) ?usize {
            if (index == 0) return null;
            return (index - 1) / 2;
        }

        /// Return the index of the grandparent node or null if none exists.
        pub fn grandparentIndex(index: usize) ?usize {
            const parent = Self.parentIndex(index) orelse return null;
            return Self.parentIndex(parent);
        }

        /// Swap the elements at indices `i` and `j`.
        fn swap(self: *Self, i: usize, j: usize) void {
            const tmp = self.data.items[i];
            self.data.items[i] = self.data.items[j];
            self.data.items[j] = tmp;
        }

        /// Bubble up the element at `index` along the min–levels.
        fn bubbleUpMin(self: *Self, index: usize) void {
            var index_ = index;

            while (true) {
                const gp = Self.grandparentIndex(index_);

                if (gp) |gpi| {
                    if (Self.key(self.data.items[index_]) < Self.key(self.data.items[gpi])) {
                        self.swap(index_, gpi);
                        index_ = gpi;
                        continue;
                    }
                }

                break;
            }
        }

        /// Bubble up the element at `index` along the max–levels.
        fn bubbleUpMax(self: *Self, index: usize) void {
            var index_ = index;

            while (true) {
                const gp = Self.grandparentIndex(index_);

                if (gp) |gpi| {
                    if (Self.key(self.data.items[index_]) > Self.key(self.data.items[gpi])) {
                        self.swap(index_, gpi);
                        index_ = gpi;
                        continue;
                    }
                }

                break;
            }
        }

        /// Insert a new value into the heap.
        pub fn insert(self: *Self, value: T) !void {
            try self.data.append(value);
            const index = self.data.items.len - 1;
            if (index == 0) return; // Only element.
            const parent = Self.parentIndex(index) orelse return;
            const lvl = Self.level(index);

            if (lvl % 2 == 0) { // Even level: min–level.
                if (Self.key(self.data.items[index]) > Self.key(self.data.items[parent])) {
                    self.swap(index, parent);
                    self.bubbleUpMax(parent);
                } else {
                    self.bubbleUpMin(index);
                }
            } else { // Odd level: max–level.
                if (Self.key(self.data.items[index]) < Self.key(self.data.items[parent])) {
                    self.swap(index, parent);
                    self.bubbleUpMin(parent);
                } else {
                    self.bubbleUpMax(index);
                }
            }
        }

        /// Return the index of the smallest child or grandchild of the node at `index`,
        /// or null if no children exist.
        fn minChildGrandchildIndex(self: *Self, index: usize) ?usize {
            const n = self.data.items.len;
            var candidate: ?usize = null;
            const left = 2 * index + 1;
            const right = 2 * index + 2;

            if (left < n) {
                candidate = left;
            }

            if (right < n) {
                if (candidate == null or Self.key(self.data.items[right]) < Self.key(self.data.items[candidate.?])) {
                    candidate = right;
                }
            }

            if (left < n) {
                const leftLeft = 2 * left + 1;
                const leftRight = 2 * left + 2;

                if (leftLeft < n) {
                    if (candidate == null or Self.key(self.data.items[leftLeft]) < Self.key(self.data.items[candidate.?])) {
                        candidate = leftLeft;
                    }
                }

                if (leftRight < n) {
                    if (candidate == null or Self.key(self.data.items[leftRight]) < Self.key(self.data.items[candidate.?])) {
                        candidate = leftRight;
                    }
                }
            }

            if (right < n) {
                const rightLeft = 2 * right + 1;
                const rightRight = 2 * right + 2;

                if (rightLeft < n) {
                    if (candidate == null or Self.key(self.data.items[rightLeft]) < Self.key(self.data.items[candidate.?])) {
                        candidate = rightLeft;
                    }
                }

                if (rightRight < n) {
                    if (candidate == null or Self.key(self.data.items[rightRight]) < Self.key(self.data.items[candidate.?])) {
                        candidate = rightRight;
                    }
                }
            }

            return candidate;
        }

        /// Push the element at `index` down along the min–levels.
        fn pushDownMin(self: *Self, index: usize) void {
            var index_ = index;

            while (true) {
                const mOpt = self.minChildGrandchildIndex(index_);
                if (mOpt == null) break;
                const m = mOpt.?;
                const isGrandchild = m >= (4 * index_ + 3);

                if (Self.key(self.data.items[m]) < Self.key(self.data.items[index_])) {
                    if (isGrandchild) {
                        self.swap(m, index_);
                        if (Self.parentIndex(m)) |p| {
                            if (Self.key(self.data.items[m]) > Self.key(self.data.items[p])) {
                                self.swap(m, p);
                            }
                        }
                        index_ = m;
                        continue;
                    } else {
                        self.swap(m, index_);
                    }
                }

                break;
            }
        }

        /// Return the index of the largest child or grandchild of the node at `index`,
        /// or null if no children exist.
        fn maxChildGrandchildIndex(self: *Self, index: usize) ?usize {
            const n = self.data.items.len;
            var candidate: ?usize = null;
            const left = 2 * index + 1;
            const right = 2 * index + 2;

            if (left < n) {
                candidate = left;
            }

            if (right < n) {
                if (candidate == null or Self.key(self.data.items[right]) > Self.key(self.data.items[candidate.?])) {
                    candidate = right;
                }
            }

            if (left < n) {
                const leftLeft = 2 * left + 1;
                const leftRight = 2 * left + 2;

                if (leftLeft < n) {
                    if (candidate == null or Self.key(self.data.items[leftLeft]) > Self.key(self.data.items[candidate.?])) {
                        candidate = leftLeft;
                    }
                }

                if (leftRight < n) {
                    if (candidate == null or Self.key(self.data.items[leftRight]) > Self.key(self.data.items[candidate.?])) {
                        candidate = leftRight;
                    }
                }
            }

            if (right < n) {
                const rightLeft = 2 * right + 1;
                const rightRight = 2 * right + 2;

                if (rightLeft < n) {
                    if (candidate == null or Self.key(self.data.items[rightLeft]) > Self.key(self.data.items[candidate.?])) {
                        candidate = rightLeft;
                    }
                }

                if (rightRight < n) {
                    if (candidate == null or Self.key(self.data.items[rightRight]) > Self.key(self.data.items[candidate.?])) {
                        candidate = rightRight;
                    }
                }
            }

            return candidate;
        }

        /// Push the element at `index` down along the max–levels.
        fn pushDownMax(self: *Self, index: usize) void {
            while (true) {
                const mOpt = self.maxChildGrandchildIndex(index);
                if (mOpt == null) break;
                const m = mOpt.?;
                const isGrandchild = m >= (4 * index + 3);

                if (Self.key(self.data.items[m]) > Self.key(self.data.items[index])) {
                    if (isGrandchild) {
                        self.swap(m, index);

                        if (Self.parentIndex(m)) |p| {
                            if (Self.key(self.data.items[m]) < Self.key(self.data.items[p])) {
                                self.swap(m, p);
                            }
                        }

                        index = m;
                        continue;
                    } else {
                        self.swap(m, index);
                    }
                }

                break;
            }
        }

        /// Remove and return the minimum element (at the root).
        pub fn deleteMin(self: *Self) !T {
            if (self.data.items.len == 0) return Error.EmptyHeap;
            const minValue = self.data.items[0];
            const lastIndex = self.data.items.len - 1;

            if (lastIndex == 0) {
                _ = self.data.pop();
                return minValue;
            }

            self.data.items[0] = self.data.items[lastIndex];
            _ = self.data.pop();
            self.pushDownMin(0);
            return minValue;
        }

        /// Remove and return the maximum element.
        pub fn deleteMax(self: *Self) !T {
            const n = self.data.items.len;
            if (n == 0) return Error.EmptyHeap;
            if (n == 1) return self.deleteMin();
            var maxIndex: usize = 1;

            if (n > 2 and Self.key(self.data.items[2]) > Self.key(self.data.items[1])) {
                maxIndex = 2;
            }

            const maxValue = self.data.items[maxIndex];
            const lastIndex = self.data.items.len - 1;
            self.data.items[maxIndex] = self.data.items[lastIndex];
            _ = self.data.pop();

            const lvl = Self.level(maxIndex);

            if (Self.parentIndex(maxIndex)) |p| {
                if (lvl % 2 == 0) {
                    if (Self.key(self.data.items[maxIndex]) > Self.key(self.data.items[p])) {
                        self.swap(maxIndex, p);
                        self.bubbleUpMax(p);
                    } else {
                        self.pushDownMin(maxIndex);
                    }
                } else {
                    if (Self.key(self.data.items[maxIndex]) < Self.key(self.data.items[p])) {
                        self.swap(maxIndex, p);
                        self.bubbleUpMin(p);
                    } else {
                        self.pushDownMax(maxIndex);
                    }
                }
            }

            return maxValue;
        }

        /// Return (without removing) the minimum element.
        pub fn findMin(self: *Self) !T {
            if (self.data.items.len == 0) return Error.EmptyHeap;
            return self.data.items[0];
        }

        /// Return (without removing) the maximum element.
        pub fn findMax(self: *Self) !T {
            const n = self.data.items.len;
            if (n == 0) return Error.EmptyHeap;
            if (n == 1) return self.data.items[0];
            if (n == 2) return self.data.items[1];
            return if (Self.key(self.data.items[1]) > Self.key(self.data.items[2]))
                self.data.items[1]
            else
                self.data.items[2];
        }
    };
}

/// Example usage of the generic MinMaxHeap with both integer and floating–point types.
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Integer Heap ===\n", .{});
    var heap_int = try MinMaxHeap(i32).init(allocator);
    defer heap_int.deinit();

    try heap_int.insert(10);
    try heap_int.insert(20);
    try heap_int.insert(5);
    try heap_int.insert(30);
    try heap_int.insert(1);
    try heap_int.insert(15);
    try heap_int.insert(25);

    std.debug.print("Min (i32): {d}\n", .{try heap_int.findMin()});
    std.debug.print("Max (i32): {d}\n", .{try heap_int.findMax()});
    _ = try heap_int.deleteMin();
    _ = try heap_int.deleteMax();
    std.debug.print("After deletion, Min (i32): {d}\n", .{try heap_int.findMin()});
    std.debug.print("After deletion, Max (i32): {d}\n\n", .{try heap_int.findMax()});

    std.debug.print("=== Floating–Point Heap ===\n", .{});
    var heap_float = try MinMaxHeap(f64).init(allocator);
    defer heap_float.deinit();

    try heap_float.insert(10.5);
    try heap_float.insert(20.5);
    try heap_float.insert(5.5);
    try heap_float.insert(30.5);
    try heap_float.insert(1.5);
    try heap_float.insert(15.5);
    try heap_float.insert(25.5);

    std.debug.print("Min (f64): {f}\n", .{try heap_float.findMin()});
    std.debug.print("Max (f64): {f}\n", .{try heap_float.findMax()});
    _ = try heap_float.deleteMin();
    _ = try heap_float.deleteMax();
    std.debug.print("After deletion, Min (f64): {f}\n", .{try heap_float.findMin()});
    std.debug.print("After deletion, Max (f64): {f}\n", .{try heap_float.findMax()});
}
