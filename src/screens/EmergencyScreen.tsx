import React, { useEffect, useState, useRef, useCallback } from "react";
import {View,Text,StyleSheet,TouchableOpacity,Alert,FlatList,ActivityIndicator,Linking,} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getEmergencyContacts } from "../services/userService";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import axios from "axios";
import { useTranslation } from "../context/TranslationContext";
import { useFocusEffect } from "@react-navigation/native";
import * as Speech from "expo-speech";
import { useSpeech } from "../hooks/useSpeech";
import * as SMS from "expo-sms";
import * as Location from "expo-location";
import { SERVER_IP } from "../config/config";
import FallDetection from "../components/fallDetection/FallDetection";

const EmergencyScreen = () => {
  const [contacts, setContacts] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<any>(null);
  const alertTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const { translateText, targetLanguage } = useTranslation();
  const speakText = useSpeech();
  const [translations, setTranslations] = useState({
    title: "Emergency Services",
    description: "Call emergency services immediately",
    call: "Call",
    fallback: "No contacts found. Please add from your Profile.",
    login: "Please log in to access emergency services.",
  });

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(getAuth(), (user) => {
      setUser(user);
      if (user) {
        fetchContacts();
      } else {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      fetchContacts();
    }, [])
  );
  const fetchContacts = async () => {
    try {
      const result = await getEmergencyContacts();
      setContacts(result);
    } catch (error) {
      console.error("Failed to fetch contacts:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const translateUI = async () => {
      try {
        const translated = {
          title: await translateText("Emergency Services"),
          description: await translateText(
            "Call emergency services immediately"
          ),
          call: await translateText("Call"),
          fallback: await translateText(
            "No contacts found. Please add from your Profile."
          ),
          login: await translateText(
            "Please log in to access emergency services."
          ),
        };
        setTranslations(translated);
      } catch (error) {
        console.error("Translation error:", error);
      }
    };

    translateUI();
  }, [targetLanguage, translateText]);


  const getCurrentLocation = async () => {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission denied", "Location permission is required");
      return null;
    }

    const location = await Location.getCurrentPositionAsync({});
    return location.coords;
  };

  const generateMapsLink = (lat: number, lon: number) => {
    return `https://www.google.com/maps?q=${lat},${lon}`;
  };

  const handleMessage = async () => {
    const coords = await getCurrentLocation();
    if (!coords) return;

    const mapLink = generateMapsLink(coords.latitude, coords.longitude);
    return mapLink;
  }
  
  const makeEmergencyCall = async (number: string) => {
    try {
      const response = await axios.post(`http://${SERVER_IP}:8000/make-call`, {
        to: number,
      });
      Alert.alert("Call Started", `Status: ${response.data.status}`);
    } catch (error) {
      console.error("Call failed:", error);
      Alert.alert("Call Failed", "Unable to place the call.");
    }
  };
 /* const sendEmergencyMessage = async (number: string, message: string) => {
    try {
      // First check if SMS is available
      const isAvailable = await SMS.isAvailableAsync();
      if (!isAvailable) {
        Alert.alert("Error", "SMS is not available on this device");
        return;
      }
      const coords = await getCurrentLocation();
      if (!coords) return;

      const mapLink = generateMapsLink(coords.latitude, coords.longitude);
      // Send the SMS
      const { result } = await SMS.sendSMSAsync(
        [number], 
        message ||
          `EMERGENCY: I need immediate help! Please contact me as soon as possible. ${mapLink}`, // Default emergency message if none provided
      );

      switch (result) {
        case "sent":
          Alert.alert("Success", "Emergency message was sent");
          break;
        case "cancelled":
          Alert.alert("Cancelled", "Message sending was cancelled");
          break;
        default:
          Alert.alert("Status", "Message status unknown");
      }
    } catch (error) {
      console.error("SMS failed:", error);
      Alert.alert("Error", "Failed to send emergency message");
    }
  };*/

  const sendEmergencyMessage = async (number: string, message: string) => {
    const coords = await getCurrentLocation();
      if (!coords) return;

      const mapLink = generateMapsLink(coords.latitude, coords.longitude);

    try {
      const response = await axios.post(`http://${SERVER_IP}:8000/send-sms`, {
        to: number,
        message: message || `EMERGENCY: I need immediate help! Please contact me as soon as possible. ${mapLink}`,
      });
      Alert.alert('Message Sent', `Status: ${response.data.status}`);
    } catch (error) {
      console.error('Message failed:', error);
      Alert.alert('Message Failed', 'Unable to send the message.');
    }
  };


  const sendWhatsAppMessage = async (number: string, message: string) => {
    const coords = await getCurrentLocation();
    if (!coords) return;

    const mapLink = generateMapsLink(coords.latitude, coords.longitude);

    try {
      const response = await axios.post(
        `http://${SERVER_IP}:8000/send-whatsapp`,
        {
          to: number,
          from: "+14155238886",
          message: message || `EMERGENCY: I need immediate help! Please contact me as soon as possible. ${mapLink}`,
        }
      );
      Alert.alert("WhatsApp Sent", `Status: ${response.data.status}`);
    } catch (error) {
      console.error("WhatsApp message failed:", error);
      Alert.alert("Failed", "Could not send WhatsApp message.");
    }
  };

  const handleFallDetected = async (): Promise<void> => {
    if (alertTimeoutRef.current) {
      clearTimeout(alertTimeoutRef.current);
    }

    const alertMessage = await translateText("Fall detected! Are you okay?");
    await speakText(alertMessage);

    Alert.alert(
      await translateText("Fall Detected!"),
      await translateText(
        "Are you okay? Automatic emergency call in 20 seconds."
      ),
      [
        {
          text: await translateText("I'm OK"),
          onPress: () => {
            if (alertTimeoutRef.current) {
              clearTimeout(alertTimeoutRef.current);
            }
          },
          style: "cancel",
        },
        {
          text: await translateText("Get Help"),
          onPress: async () => {
            if (alertTimeoutRef.current) {
              clearTimeout(alertTimeoutRef.current);
            }
            if (contacts.length > 0) {
              await speakText(
                await translateText(
                  "Calling, sending SMS, and WhatsApp to emergency contacts now"
                )
              );

              const translatedMessage = await translateText(
                ""
              );

              const callPromises = contacts.map((contact) =>
                makeEmergencyCall(contact)
              );
              const smsPromises = contacts.map((contact) =>
                sendEmergencyMessage(contact, translatedMessage)
              );
              const whatsappPromises = contacts.map((contact) =>
                sendWhatsAppMessage(contact, translatedMessage)
              );

              await Promise.all([
                ...callPromises,
                ...smsPromises,
                ...whatsappPromises,
              ]);
            }
          },
          style: "destructive",
        },
      ],
      { cancelable: false }
    );

    alertTimeoutRef.current = setTimeout(async () => {
      const noResponseMessage = await translateText(
        "No response detected. Calling emergency contact."
      );
      await speakText(noResponseMessage);
    
      const translatedMessage = "";
    
      const callPromises = contacts.map((contact) =>
        makeEmergencyCall(contact)
      );

      const smsPromises = contacts.map((contact) =>
        sendEmergencyMessage(contact, translatedMessage)
      );
    
      const whatsappPromises = contacts.map((contact) =>
        sendWhatsAppMessage(contact, translatedMessage)
      );
    
      await Promise.all([...callPromises, ...smsPromises, ...whatsappPromises]);
    }, 8000); // 8 seconds
  };
    

  useEffect(() => {
    return () => {
      if (alertTimeoutRef.current) {
        clearTimeout(alertTimeoutRef.current);
      }
    };
  }, []);

  if (loading) {
    return (
      <ActivityIndicator size="large" color="#000" style={styles.loader} />
    );
  }

  if (!user) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>{translations.login}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FallDetection onFallDetected={handleFallDetected} />
      <Text style={styles.title}>{translations.title}</Text>
      <Text style={styles.description}>{translations.description}</Text>

      {contacts.length === 0 ? (
        <Text style={styles.fallback}>{translations.fallback}</Text>
      ) : (
        <FlatList
          data={contacts}
          keyExtractor={(item, index) => index.toString()}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.serviceButton}
              onPress={() => makeEmergencyCall(item)}
              onLongPress={() => {
                Alert.alert(
                  "Choose Action",
                  `What would you like to do with ${item}?`,
                  [
                    { text: "Call", onPress: () => makeEmergencyCall(item) },
                    {
                      text: "Send Emergency SMS",
                      onPress: () =>
                        Alert.alert(
                          "Select Message",
                          "Choose an emergency message:",
                          [
                            {
                              text: "Need Help!",
                              onPress: () =>
                                sendEmergencyMessage(
                                  item,
                                  "EMERGENCY: I need help! Please contact me ASAP!"
                                ),
                            },
                            {
                              text: "Medical Emergency",
                              onPress: () =>
                                sendEmergencyMessage(
                                  item,
                                  "MEDICAL EMERGENCY: Need immediate assistance! Please help!"
                                ),
                            },
                            {
                              text: "I've Fallen",
                              onPress: () =>
                                sendEmergencyMessage(
                                  item,
                                  "EMERGENCY: I've fallen and need assistance! Please help!"
                                ),
                            },
                            {
                              text: "Custom Message",
                              onPress: () => sendEmergencyMessage(item, ""),
                            },
                            {
                              text: "Cancel",
                              style: "cancel",
                            },
                          ]
                        ),
                    },
                    {
                      text: "WhatsApp",
                      onPress: () =>
                        sendWhatsAppMessage(item, "Hello, how are you?"),
                    },
                    { text: "Cancel", style: "cancel" },
                  ]
                );
              }}
            >
              <Ionicons name="call-outline" size={28} color="#007AFF" />
              <Text style={styles.serviceText}>{item}</Text>
              <Text style={styles.callText}>{translations.call}</Text>
            </TouchableOpacity>
          )}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#E6E6FA",
  },
  title: {
      fontSize: 30,
      fontWeight: "bold",
      margin: 16,
  },
  description: {
    fontSize: 16,
    color: "#666",
    marginBottom: 20,
    textAlign: "center",
  },
  fallback: {
    fontSize: 16,
    color: "gray",
    textAlign: "center",
    marginTop: 20,
  },
  serviceButton: {
    backgroundColor: "#fff",
    width: 300,
    padding: 15,
    marginBottom: 15,
    borderRadius: 10,
    alignItems: "center",
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  serviceText: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
    marginTop: 8,
  },
  callText: {
    fontSize: 14,
    color: "#007AFF",
    marginTop: 4,
  },
  loader: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});

export default EmergencyScreen;
