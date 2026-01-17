import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  FlatList,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, useRoute } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';

const Stack = createNativeStackNavigator();
const API_BASE = 'http://134.209.165.83:8080'


const probabilityLabel = (p) => {
  if (p >= 0.7) return 'Likely Interaction';
  if (p >= 0.4) return 'Uncertain Interaction';
  return 'Unlikely Interaction';
};

function SearchScreen({ navigation }) {
  const route = useRoute();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const [lastResetKey, setLastResetKey] = useState(null);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setLoading(false);
      return () => {};
    }

    let isActive = true;
    const controller = new AbortController();
    const debounceId = setTimeout(async () => {
      setLoading(true);
      setError('');
      try {
        const resp = await fetch(`${API_BASE}/v1/drugs/search?q=${encodeURIComponent(query.trim())}`, {
          signal: controller.signal,
        });
        if (!resp.ok) throw new Error(`search failed: ${resp.status}`);
        const data = await resp.json();
        if (isActive) setResults(data.results ?? []);
      } catch (err) {
        if (isActive && err.name !== 'AbortError') {
          setError('Unable to fetch search results right now.');
          setResults([]);
        }
      } finally {
        if (isActive) setLoading(false);
      }
    }, 350);

    return () => {
      isActive = false;
      controller.abort();
      clearTimeout(debounceId);
    };
  }, [query]);

  const toggleSelect = (name) => {
    setSelected((prev) => {
      if (prev.includes(name)) {
        return prev.filter((n) => n !== name);
      }
      if (prev.length >= 2) {
        return [prev[1], name];
      }
      return [...prev, name];
    });
    setQuery("")
  };

  const removeSelected = (name) => {
    setSelected((prev) => prev.filter((n) => n !== name));
  };

  useEffect(() => {
    const resetKey = route.params?.resetSelectionKey;
    if (resetKey && resetKey !== lastResetKey) {
      setSelected([]);
      setQuery('');
      setResults([]);
      setError('');
      setLastResetKey(resetKey);
    }
  }, [route.params?.resetSelectionKey, lastResetKey]);
  

  const handleCheck = async () => {
    if (selected.length !== 2) return;
    setChecking(true);
    setError('');
    try {
      const resp = await fetch(`${API_BASE}/v1/ddi/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drugAName: selected[0], drugBName: selected[1] }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || 'prediction failed');
      }
      const data = await resp.json();
      navigation.navigate('Result', { result: data });
    } catch (err) {
      setError('Could not retrieve interaction data. Try again.');
    } finally {
      setChecking(false);
    }
  };

  const renderItem = ({ item }) => {
    const isSelected = selected.includes(item);
    return (
      <Pressable
        onPress={() => toggleSelect(item)}
        style={[styles.resultItem, isSelected && styles.resultItemSelected]}
      >
        <Text style={[styles.resultText, isSelected && styles.resultTextSelected]}>{item}</Text>
        {isSelected && <Text style={styles.resultBadge}>Selected</Text>}
      </Pressable>
    );
  };

  return (
    <SafeAreaView style={styles.screenContainer}>
      <View style={styles.screen}>
        <StatusBar style="dark" />
        <Text style={styles.heading}>Drug Interaction Checker</Text>
        <TextInput
          value={query}
          onChangeText={setQuery}
          placeholder="Search for a drug"
          style={styles.searchInput}
          autoCapitalize="none"
          placeholderTextColor={"#bcbcbcff"}
          autoCorrect={false}
          returnKeyType="search"
        />
        <View style={styles.selectedRow}>
          {selected.map((name) => (
            <Pressable
              key={name}
              onPress={() => removeSelected(name)}
              hitSlop={8}
              style={styles.selectedChip}
            >
              <Text style={styles.selectedChipText}>{name}</Text>
              <Text style={styles.selectedChipRemove}>×</Text>
            </Pressable>
          ))}
        </View>
        {loading && <ActivityIndicator size="small" color="#2563eb" style={styles.loading} />}
        {error !== '' && <Text style={styles.errorText}>{error}</Text>}
        <FlatList
          data={results}
          keyExtractor={(item) => item}
          renderItem={renderItem}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.listContent}
          ListEmptyComponent={
            !loading && query.trim() !== '' ? (
              <Text style={styles.emptyText}>No matches found.</Text>
            ) : null
          }
        />
        <Pressable
          disabled={selected.length !== 2 || checking}
          onPress={handleCheck}
          style={({ pressed }) => [
            styles.checkButton,
            (pressed || checking) && styles.checkButtonPressed,
            selected.length !== 2 && styles.checkButtonDisabled,
          ]}
        >
          {checking ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.checkButtonText}>
              {selected.length === 2 ? 'Check Interaction' : 'Select two drugs'}
            </Text>
          )}
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

function ResultScreen({ route, navigation }) {
  const { result } = route.params;
  const { drugAName, drugBName, probability, label, smilesA, smilesB } = result;
  const progress = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    progress.setValue(0);
    Animated.timing(progress, {
      toValue: probability,
      duration: 900,
      useNativeDriver: false,
    }).start();
  }, [probability, progress]);

  const progressWidth = progress.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });
  const progressColor = progress.interpolate({
    inputRange: [0, 0.4, 0.7, 1],
    outputRange: ['#10b981', '#facc15', '#fb923c', '#ef4444'],
  });

  const score = useMemo(() => Math.round(probability * 1000) / 10, [probability]);

  const handleReset = () => {
    navigation.reset({
      index: 0,
      routes: [{ name: 'Search', params: { resetSelectionKey: Date.now() } }],
    });
  };

  return (
    <SafeAreaView style={styles.screenContainer}>
      <View style={styles.screen}>
        <StatusBar style="light" />
        <Text style={styles.heading}>Interaction Result</Text>
        <View style={styles.resultCard}>
        <Text style={styles.resultTitle}>{drugAName}</Text>
        <Text style={styles.resultSubtitle}>{smilesA}</Text>
        <View style={styles.resultDivider} />
        <Text style={styles.resultTitle}>{drugBName}</Text>
        <Text style={styles.resultSubtitle}>{smilesB}</Text>
      </View>
        <View style={styles.progressContainer}>
        <View style={styles.progressTrack}>
          <Animated.View
            style={[styles.progressFill, { width: progressWidth, backgroundColor: progressColor }]}
          />
        </View>
        <Animated.Text style={styles.progressLabel}>{score}%</Animated.Text>
        <Text style={styles.progressDescription}>{probabilityLabel(probability)}</Text>
        <Text style={styles.progressCaption}>Model label: {label.replace('ddi_', '').toUpperCase()}</Text>
        </View>
        <Pressable onPress={handleReset} style={styles.backButton}>
        <Text style={styles.backButtonText}>Check another pair</Text>
      </Pressable>
      </View>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Search" component={SearchScreen} options={{ headerShown: false }} />
          <Stack.Screen name="Result" component={ResultScreen} options={{ headerShown: false }} />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  screenContainer: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  screen: {
    flex: 1,
    backgroundColor: '#0f172a',
    paddingHorizontal: 24,
    paddingTop: 24,
    paddingBottom: 24,
  },
  heading: {
    fontSize: 24,
    fontWeight: '700',
    color: '#f8fafc',
    marginBottom: 16,
  },
  searchInput: {
    backgroundColor: '#1e293b',
    color: '#f8fafc',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  selectedRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
  },
  selectedChip: {
    backgroundColor: '#38bdf8',
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  selectedChipText: {
    color: '#0f172a',
    fontWeight: '600',
    marginRight: 4,
  },
  selectedChipRemove: {
    color: '#0f172a',
    fontSize: 16,
    fontWeight: '700',
    paddingHorizontal: 4,
    marginLeft: 4,
  },
  loading: {
    marginTop: 12,
  },
  errorText: {
    color: '#f87171',
    marginTop: 12,
  },
  listContent: {
    paddingVertical: 16,
  },
  emptyText: {
    color: '#cbd5f5',
    textAlign: 'center',
    marginTop: 24,
  },
  resultItem: {
    backgroundColor: '#1e293b',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 10,
  },
  resultItemSelected: {
    borderColor: '#38bdf8',
    backgroundColor: '#0f172a',
  },
  resultText: {
    color: '#e2e8f0',
    fontSize: 16,
    fontWeight: '500',
  },
  resultTextSelected: {
    color: '#38bdf8',
  },
  resultBadge: {
    marginTop: 6,
    color: '#38bdf8',
    fontSize: 12,
  },
  checkButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 24,
  },
  checkButtonPressed: {
    opacity: 0.8,
  },
  checkButtonDisabled: {
    backgroundColor: '#1e3a8a',
  },
  checkButtonText: {
    color: '#f8fafc',
    fontWeight: '700',
    fontSize: 16,
  },
  resultCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#334155',
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#f8fafc',
  },
  resultSubtitle: {
    fontSize: 13,
    color: '#94a3b8',
    marginTop: 4,
  },
  resultDivider: {
    height: 1,
    backgroundColor: '#334155',
    marginVertical: 12,
  },
  progressContainer: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 24,
  },
  progressTrack: {
    height: 18,
    backgroundColor: '#0f172a',
    borderRadius: 12,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 12,
  },
  progressLabel: {
    marginTop: 14,
    fontSize: 28,
    color: '#f8fafc',
    fontWeight: '800',
    textAlign: 'center',
  },
  progressDescription: {
    textAlign: 'center',
    color: '#cbd5f5',
    marginTop: 6,
    fontSize: 16,
    fontWeight: '600',
  },
  progressCaption: {
    textAlign: 'center',
    color: '#94a3b8',
    marginTop: 4,
  },
  backButton: {
    backgroundColor: '#38bdf8',
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#0f172a',
    fontWeight: '700',
    fontSize: 16,
  },
});
