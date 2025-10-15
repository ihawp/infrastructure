-- Database: whatever
-- Created from Excalidraw schema design

-- Create database
CREATE DATABASE IF NOT EXISTS whatever;
USE whatever;

-- Accounts Table
-- Main user accounts table with authentication and profile information
CREATE TABLE accounts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    username VARCHAR(255) NOT NULL UNIQUE,
    pfp_url VARCHAR(500) NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    email_verified_at DATETIME NULL,
    account_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_email_verified (email_verified),
    INDEX idx_account_created (account_created)
);

-- Refresh Tokens Table
-- Stores refresh tokens for maintaining user sessions
CREATE TABLE refresh_tokens (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    token_hash VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    used_at DATETIME NULL,
    revoked BOOLEAN NOT NULL DEFAULT FALSE,
    
    FOREIGN KEY (user_id) REFERENCES accounts(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_token_hash (token_hash),
    INDEX idx_expires_at (expires_at),
    INDEX idx_revoked (revoked)
);

-- Additional indexes for performance
CREATE INDEX idx_accounts_email_username ON accounts(email, username);
CREATE INDEX idx_refresh_tokens_user_expires ON refresh_tokens(user_id, expires_at);

-- Clean up expired tokens (can be run periodically)
-- DELETE FROM refresh_tokens WHERE expires_at < NOW() AND revoked = TRUE;
-- DELETE FROM magic_keys WHERE expires_at < NOW() AND revoked = TRUE;
