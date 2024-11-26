const std = @import("std");

pub fn SkipList(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            key: u64,
            value: T,
            next: ?Node = null,
            down: ?Node = null,
        };

        head: ?Node = null,
        tail: ?Node = null,
        len: u64 = 0,

        fn init() SkipList {
            return SkipList{
                .head = null,
                .tail = null,
                .len = 0,
            };
        }

        fn deinit(self: *Self) void {
            var current: ?Node = self.head;
            while (current != null) {
                const next = current.next;
                std.mem.free(current);
                current = next;
            }
        }

        fn insert(self: *Self, key: u64, value: T) void {
            var current: ?Node = self.head;
            var stack: [16]Node = undefined;
            var stack_len: usize = 0;

            while (current != null) {
                if (current.next != null and current.next.key <= key) {
                    current = current.next;
                    continue;
                }

                if (stack_len < 16) {
                    stack[stack_len] = current;
                    stack_len += 1;
                }

                if (current.down == null) {
                    const new_node = Node{ .key = key, .value = value, .next = current.next };
                    current.next = &new_node;
                    if (self.tail == null or key > self.tail.key) {
                        self.tail = &new_node;
                    }
                    self.len += 1;
                    break;
                }

                current = current.down;
            }

            while (stack_len > 0) {
                stack_len -= 1;
                const prev = &stack[stack_len];
                const new_node = Node{ .key = key, .value = value, .next = prev.next, .down = null };
                prev.next = &new_node;

                if (self.tail == null or key > self.tail.key) {
                    self.tail = &new_node;
                }

                self.len += 1;

                if (stack_len > 0) {
                    new_node.down = &stack[stack_len - 1];
                }
            }
        }

        fn find(self: *Self, key: u64) ?T {
            var current: ?Node = self.head;
            while (current != null) {
                if (current.next == null or current.next.key > key) {
                    if (current.down == null) {
                        return null;
                    }
                    current = current.down;
                } else if (current.next.key == key) {
                    return current.next.value;
                } else {
                    current = current.next;
                }
            }
            return null;
        }
    };
}
