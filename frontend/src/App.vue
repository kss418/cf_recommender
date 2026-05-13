<script setup>
import { computed, ref } from "vue";

const API_BASE_URL = "http://127.0.0.1:8000";

const handle = ref("");
const analysis = ref(null);
const selectedTag = ref(null);
const loading = ref(false);
const error = ref("");

const tagSkills = computed(() => analysis.value?.tag_skills ?? []);
const selectedProblems = computed(() => {
  if (!selectedTag.value || !analysis.value?.problems_by_tag) {
    return [];
  }

  return analysis.value.problems_by_tag[selectedTag.value] ?? [];
});

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${Math.round(Number(value) * 100)}%`;
}

function getTagBarWidth(tagSkill) {
  if (!tagSkills.value.length) {
    return 0;
  }

  const ratings = tagSkills.value.map((tag) => tag.skill_rating);
  const minRating = Math.min(...ratings);
  const maxRating = Math.max(...ratings);
  if (maxRating === minRating) {
    return 100;
  }

  return 100 - ((tagSkill.skill_rating - minRating) / (maxRating - minRating)) * 72;
}

async function analyzeHandle() {
  const trimmedHandle = handle.value.trim();
  if (!trimmedHandle) {
    return;
  }

  loading.value = true;
  error.value = "";
  analysis.value = null;
  selectedTag.value = null;

  try {
    const response = await fetch(`${API_BASE_URL}/recommend/${trimmedHandle}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    analysis.value = payload;
    selectedTag.value = payload.tag_skills?.[0]?.tag ?? null;
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <main class="app-shell">
    <header class="top-bar">
      <div>
        <p class="eyebrow">Codeforces Recommender</p>
        <h1>Weakness Map</h1>
      </div>
      <form class="search-form" @submit.prevent="analyzeHandle">
        <label for="handle">Handle</label>
        <div class="search-row">
          <input
            id="handle"
            v-model="handle"
            autocomplete="off"
            spellcheck="false"
          />
          <button type="submit" :disabled="loading">
            {{ loading ? "Analyzing" : "Analyze" }}
          </button>
        </div>
      </form>
    </header>

    <div v-if="error" class="error-banner">{{ error }}</div>

    <section v-if="analysis" class="workspace">
      <aside class="summary-panel">
        <div class="metric accent">
          <span>Handle</span>
          <strong>{{ analysis.user_id }}</strong>
        </div>
        <div class="metric">
          <span>Submissions</span>
          <strong>{{ analysis.user_data?.submission_count ?? "-" }}</strong>
        </div>
        <div class="metric">
          <span>AC</span>
          <strong>{{ analysis.user_data?.accepted_problem_count ?? "-" }}</strong>
        </div>
      </aside>

      <section class="chart-panel">
        <div class="panel-header">
          <div>
            <h2>Tag Skill</h2>
            <p>lower estimated skill first</p>
          </div>
          <span>{{ tagSkills.length }} tags</span>
        </div>
        <div class="tag-legend">
          <span>Tag</span>
          <span>Relative weakness</span>
          <span>Estimated skill</span>
          <span>Used / AC / confidence</span>
        </div>
        <div class="tag-list">
          <button
            v-for="tagSkill in tagSkills"
            :key="tagSkill.tag"
            :class="tagSkill.tag === selectedTag ? 'tag-row selected' : 'tag-row'"
            type="button"
            @click="selectedTag = tagSkill.tag"
          >
            <span class="tag-name">{{ tagSkill.tag }}</span>
            <span class="bar-track">
              <span
                class="bar-fill"
                :style="{ width: `${getTagBarWidth(tagSkill)}%` }"
              />
            </span>
            <span class="tag-rating">
              {{ formatNumber(tagSkill.skill_rating, 0) }}
            </span>
            <span class="tag-meta">
              {{ tagSkill.exposure }} used / {{ tagSkill.solved_count }} AC /
              {{ formatPercent(tagSkill.confidence) }}
            </span>
          </button>
        </div>
      </section>

      <section class="problem-panel">
        <div class="panel-header">
          <div>
            <h2>{{ selectedTag ?? "Problems" }}</h2>
            <p>Top 10 recommendation candidates</p>
          </div>
        </div>
        <div class="problem-list">
          <a
            v-for="problem in selectedProblems"
            :key="`${problem.contest_id}${problem.index}`"
            class="problem-item"
            :href="problem.url"
            rel="noreferrer"
            target="_blank"
          >
            <div class="problem-main">
              <div class="problem-title">
                {{ problem.contest_id }}{{ problem.index }} · {{ problem.name }}
              </div>
              <div class="problem-stats">
                <div>
                  <span>Rating</span>
                  <strong>{{ problem.rating }}</strong>
                </div>
                <div>
                  <span>Solve probability</span>
                  <strong>{{ formatPercent(problem.solve_probability) }}</strong>
                </div>
                <div>
                  <span>Recommendation score</span>
                  <strong>{{ formatNumber(problem.score, 3) }}</strong>
                </div>
              </div>
              <div class="problem-tags">
                <span v-for="tag in problem.tags" :key="tag">{{ tag }}</span>
              </div>
            </div>
          </a>
          <div v-if="!selectedProblems.length" class="empty-state">
            No candidates
          </div>
        </div>
      </section>
    </section>

    <section v-else class="start-screen" aria-hidden="true">
      <div class="start-summary">
        <span class="summary-skeleton large"></span>
        <span class="summary-skeleton"></span>
        <span class="summary-skeleton"></span>
      </div>

      <div class="start-analysis">
        <div class="start-panel-header">
          <span class="skeleton-title"></span>
          <span class="skeleton-pill"></span>
        </div>
        <div class="start-tag-list">
          <span class="start-tag-row strong"></span>
          <span class="start-tag-row"></span>
          <span class="start-tag-row medium"></span>
          <span class="start-tag-row"></span>
          <span class="start-tag-row short"></span>
          <span class="start-tag-row"></span>
        </div>
      </div>

      <div class="start-problems">
        <div class="start-panel-header">
          <span class="skeleton-title short"></span>
        </div>
        <div class="start-problem-list">
          <span class="start-problem-card"></span>
          <span class="start-problem-card compact"></span>
          <span class="start-problem-card"></span>
        </div>
      </div>
    </section>
  </main>
</template>
