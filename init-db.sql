CREATE DATABASE mnist_app;
\c mnist_app;

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    predicted_digit INTEGER,
    true_label INTEGER,
    confidence FLOAT
);

-- Create index on timestamp for faster queries
CREATE INDEX idx_timestamp ON predictions(timestamp);

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mnist_app TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;