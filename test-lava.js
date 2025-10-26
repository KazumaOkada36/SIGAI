// Load environment variables
require('dotenv').config({ path: '.env.local' });

async function testLava() {
  // 1. Build the URL - routes to OpenAI via Lava
  const url = `${process.env.LAVA_BASE_URL}/forward?u=https://api.openai.com/v1/chat/completions`;

  // 2. Set up authentication
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${process.env.LAVA_FORWARD_TOKEN}`
  };

  // 3. Define the request body (standard OpenAI format)
  const requestBody = {
    model: 'gpt-4o-mini',
    messages: [
      { role: 'user', content: 'Say hello in one sentence.' }
    ]
  };

  // 4. Make the request
  const response = await fetch(url, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(requestBody)
  });

  // 5. Parse and log the response
  const data = await response.json();
  console.log('\nResponse from OpenAI:');
  console.log(JSON.stringify(data, null, 2));

  // 6. Show the Lava request ID for tracking
  const requestId = response.headers.get('x-lava-request-id');
  console.log('\nLava request ID:', requestId);
}

testLava();