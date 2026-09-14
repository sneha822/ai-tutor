// The AI's emotions: a label and colour for each tag it can start a reply with ([happy] ...). Shared by the page
// (mood chip, message labels) and the 3D avatar, which adds the face shape for each one.

export const EMOTIONS = {
  neutral: { label: "Neutral", color: "#7ee8ff" },
  happy: { label: "Happy", color: "#ffd35c" },
  excited: { label: "Excited", color: "#ff5fd2" },
  proud: { label: "Proud", color: "#57f287" },
  playful: { label: "Playful", color: "#ffb86b" },
  caring: { label: "Caring", color: "#ff9fb8" },
  thoughtful: { label: "Thoughtful", color: "#8fa8ff" },
  confused: { label: "Confused", color: "#c77dff" },
  surprised: { label: "Surprised", color: "#e9f3ff" },
  sad: { label: "Sad", color: "#4f6bff" },
  annoyed: { label: "Annoyed", color: "#ff7a2f" },
  angry: { label: "Angry", color: "#ff3b3b" },
};

export const emotionMeta = (name) => EMOTIONS[name] || EMOTIONS.neutral;
