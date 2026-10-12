package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

const (
	defaultModel = "openai/gpt-oss-20b"
	maxHistory   = 20 // keep only the last N messages per user
	reqTimeout   = 60 * time.Second
)

type ChatRequest struct {
	UserID  string `json:"user_id"`
	Message string `json:"message"`
}

type ChatResponse struct {
	Reply string `json:"reply"`
}

var (
	chatHistory = make(map[string][]openai.ChatCompletionMessage)
	mutex       sync.Mutex
)

// addUserMessage stores the user's message and returns a safe copy of the
// (trimmed) history to send to the model.
func addUserMessage(userID, text string) []openai.ChatCompletionMessage {
	mutex.Lock()
	defer mutex.Unlock()

	h := append(chatHistory[userID], openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: text,
	})
	if len(h) > maxHistory {
		h = h[len(h)-maxHistory:]
		for len(h) > 1 && h[0].Role != openai.ChatMessageRoleUser {
			h = h[1:]
		}
	}
	chatHistory[userID] = h
	return append([]openai.ChatCompletionMessage(nil), h...)
}

func addAssistantMessage(userID, text string) {
	mutex.Lock()
	defer mutex.Unlock()
	chatHistory[userID] = append(chatHistory[userID], openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: text,
	})
}

// removeLastUserMessage rolls back a user message when the model call fails,
// so history never contains a user message without a reply.
func removeLastUserMessage(userID string) {
	mutex.Lock()
	defer mutex.Unlock()
	h := chatHistory[userID]
	if n := len(h); n > 0 && h[n-1].Role == openai.ChatMessageRoleUser {
		chatHistory[userID] = h[:n-1]
	}
}

func decodeChatRequest(w http.ResponseWriter, r *http.Request) (ChatRequest, bool) {
	var req ChatRequest
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return req, false
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return req, false
	}
	if req.UserID == "" || req.Message == "" {
		http.Error(w, "user_id and message required", http.StatusBadRequest)
		return req, false
	}
	return req, true
}

func sseSend(w http.ResponseWriter, f http.Flusher, event string, payload any) {
	data, _ := json.Marshal(payload)
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	f.Flush()
}

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		log.Fatal("GROQ_API_KEY not set")
	}

	// Model name comes from the environment so a deprecation needs no code change.
	model := os.Getenv("GROQ_MODEL")
	if model == "" {
		model = defaultModel
	}

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.groq.com/openai/v1"
	client := openai.NewClientWithConfig(config)

	mux := http.NewServeMux()

	// Serve frontend
	mux.Handle("/", http.FileServer(http.Dir("./static")))

	// Standard endpoint: returns the full reply as JSON (frontend keeps working).
	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		req, ok := decodeChatRequest(w, r)
		if !ok {
			return
		}

		messages := addUserMessage(req.UserID, req.Message)

		ctx, cancel := context.WithTimeout(r.Context(), reqTimeout)
		defer cancel()

		resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    model,
			Messages: messages,
		})
		if err != nil {
			log.Println("Groq error:", err)
			removeLastUserMessage(req.UserID)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(resp.Choices) == 0 {
			log.Println("Groq returned no choices")
			removeLastUserMessage(req.UserID)
			http.Error(w, "Empty response from model", http.StatusInternalServerError)
			return
		}

		reply := resp.Choices[0].Message.Content
		addAssistantMessage(req.UserID, reply)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Reply: reply})
	})

	// Streaming endpoint: sends the reply token by token using Server-Sent Events.
	mux.HandleFunc("/chat/stream", func(w http.ResponseWriter, r *http.Request) {
		req, ok := decodeChatRequest(w, r)
		if !ok {
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		messages := addUserMessage(req.UserID, req.Message)

		// r.Context() is cancelled if the browser disconnects, which stops the Groq call.
		ctx, cancel := context.WithTimeout(r.Context(), reqTimeout)
		defer cancel()

		stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    model,
			Messages: messages,
		})
		if err != nil {
			log.Println("Groq stream error:", err)
			removeLastUserMessage(req.UserID)
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer stream.Close()

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no") // stop proxies from buffering the stream

		var full strings.Builder
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				log.Println("Groq stream recv error:", err)
				removeLastUserMessage(req.UserID)
				sseSend(w, flusher, "error", map[string]string{"error": err.Error()})
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			token := chunk.Choices[0].Delta.Content
			if token == "" {
				continue
			}
			full.WriteString(token)
			sseSend(w, flusher, "", map[string]string{"token": token})
		}

		addAssistantMessage(req.UserID, full.String())
		sseSend(w, flusher, "done", map[string]bool{"done": true})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// No WriteTimeout on purpose: it would cut long streaming responses.
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Println("🚀 Server running on port", port)
	log.Fatal(srv.ListenAndServe())
}
