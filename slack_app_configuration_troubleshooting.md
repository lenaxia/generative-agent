# Slack App Configuration Troubleshooting Guide

## Issue Identified

The Slack handler is working correctly (WebSocket connected, event handlers registered), but **no events are being received from Slack**. This indicates a **Slack app configuration issue**, not a code problem.

## Root Cause

Slack is not sending events to your bot because the Slack app is not properly configured to:

1. Subscribe to the required events
2. Have the correct OAuth scopes
3. Be installed in the workspace with proper permissions

## Required Slack App Configuration

### 1. OAuth Scopes (OAuth & Permissions)

Your Slack app needs these **Bot Token Scopes**:

```
app_mentions:read    - To receive app mention events
channels:history     - To read messages in channels
groups:history       - To read messages in private channels
im:history          - To read direct messages
mpim:history        - To read group direct messages
chat:write          - To send messages
```

### 2. Event Subscriptions (Event Subscriptions)

Enable **Event Subscriptions** and subscribe to these **Bot Events**:

```
app_mention         - When your bot is mentioned
message.channels    - Messages in channels (if needed)
message.groups      - Messages in private channels (if needed)
message.im          - Direct messages (if needed)
```

### 3. Socket Mode (Socket Mode)

- **Enable Socket Mode**: This allows your bot to receive events via WebSocket
- **Generate App-Level Token**: With `connections:write` scope (this is your `SLACK_APP_TOKEN`)

### 4. App Installation

- **Install the app** to your workspace
- **Invite the bot** to the channel where you're testing
- **Grant permissions** when prompted

## Verification Steps

### Step 1: Check Slack App Dashboard

1. Go to https://api.slack.com/apps
2. Select your app
3. Verify the configurations above

### Step 2: Check Bot Installation

1. In Slack, go to the channel where you're testing
2. Type `/invite @your-bot-name` to invite the bot
3. The bot should appear in the channel member list

### Step 3: Check Event Delivery

1. In your Slack app dashboard, go to "Event Subscriptions"
2. Check if events are being delivered (you should see delivery attempts)
3. If no delivery attempts, the issue is with event subscription configuration

### Step 4: Test Bot Mention

1. In a channel where the bot is present, type: `@your-bot-name hello`
2. You should now see debug logs in your application

## Common Issues & Solutions

### Issue: "App not responding to mentions"

**Solution**:

- Ensure bot is invited to the channel
- Check that `app_mentions:read` scope is granted
- Verify `app_mention` event subscription is enabled

### Issue: "No events in Event Subscriptions dashboard"

**Solution**:

- Enable Socket Mode
- Ensure App-Level Token has `connections:write` scope
- Verify bot token has required scopes

### Issue: "Bot appears offline"

**Solution**:

- Check that Socket Mode is enabled
- Verify both `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` are correct
- Ensure app is installed in the workspace

## Expected Behavior After Fix

Once properly configured, you should see these logs when mentioning the bot:

```
🔍 DEBUG: Received Slack event: app_mention - {event data}
🔔 Received app mention: {event data}
📢 Processing app mention from user {user}: {text}
📤 Adding app mention to message queue: {message data}
✅ App mention successfully queued. Queue size: 1
📨 Processing queued message from slack: {message}
```

## Next Steps

1. **Configure your Slack app** using the settings above
2. **Reinstall the app** to your workspace if needed
3. **Invite the bot** to your test channel
4. **Test app mentions** - you should now see the debug logs

The code fix we implemented (queue consistency) is working correctly. The issue is purely with Slack app configuration.
