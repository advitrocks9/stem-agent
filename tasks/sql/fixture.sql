-- Small fixture used by the sql task class. Three tables, deliberately
-- diverse data so questions can range from simple counts to multi-join
-- aggregations.

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

INSERT INTO users (id, name, country, signup_date) VALUES
    (1, 'Alice', 'UK', '2024-01-15'),
    (2, 'Bob', 'US', '2024-02-03'),
    (3, 'Charlie', 'UK', '2024-02-20'),
    (4, 'Diana', 'DE', '2024-03-10'),
    (5, 'Eli', 'US', '2024-04-01'),
    (6, 'Fatima', 'UK', '2024-05-22'),
    (7, 'Gunnar', 'DE', '2024-06-08'),
    (8, 'Hana', 'JP', '2024-07-14'),
    (9, 'Iris', 'UK', '2024-12-20');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Notebook', 'stationery', 4.50),
    (2, 'Pen pack', 'stationery', 7.00),
    (3, 'Desk lamp', 'lighting', 25.00),
    (4, 'Bulb 4-pack', 'lighting', 12.00),
    (5, 'Keyboard', 'electronics', 65.00),
    (6, 'Mouse', 'electronics', 22.00),
    (7, 'Headphones', 'electronics', 89.00),
    (8, 'Cable USB-C', 'electronics', 9.50),
    (9, 'Coffee beans', 'grocery', 14.00),
    (10, 'Tea box', 'grocery', 8.50),
    (11, 'Olive oil', 'grocery', 11.00),
    (12, 'Sparkling water 12-pack', 'grocery', 6.50),
    (13, 'Eraser', 'stationery', 1.20),
    (14, 'Sticky notes', 'stationery', 3.50);

INSERT INTO orders (id, user_id, product_id, quantity, order_date) VALUES
    (1, 1, 5, 1, '2024-03-02'),
    (2, 1, 6, 1, '2024-03-02'),
    (3, 2, 1, 5, '2024-03-15'),
    (4, 2, 9, 2, '2024-03-15'),
    (5, 3, 7, 1, '2024-04-04'),
    (6, 4, 3, 2, '2024-04-12'),
    (7, 5, 5, 1, '2024-05-01'),
    (8, 5, 7, 1, '2024-05-01'),
    (9, 5, 8, 3, '2024-05-01'),
    (10, 6, 9, 1, '2024-06-08'),
    (11, 6, 10, 2, '2024-06-08'),
    (12, 6, 11, 1, '2024-06-08'),
    (13, 7, 4, 4, '2024-07-02'),
    (14, 7, 12, 6, '2024-07-02'),
    (15, 8, 9, 1, '2024-08-12'),
    (16, 8, 7, 1, '2024-08-12'),
    (17, 1, 9, 3, '2024-09-01'),
    (18, 1, 11, 1, '2024-09-01'),
    (19, 4, 5, 1, '2024-09-14'),
    (20, 2, 3, 1, '2024-10-05'),
    (21, 3, 1, 10, '2024-10-20'),
    (22, 6, 7, 1, '2024-11-03'),
    (23, 2, 7, 1, '2024-11-15'),
    (24, 5, 9, 4, '2024-11-30'),
    (25, 7, 5, 1, '2024-12-01');
