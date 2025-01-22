import React, { useState } from 'react';
import { Github, Mail, ArrowRight, ArrowLeft } from 'lucide-react';
import ChatInterface from './ChatInterface';

const LandingPage = () => {
  const [showChat, setShowChat] = useState(false);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {showChat ? (
        <div className="h-screen flex flex-col">
          <div className="bg-black p-4 flex items-center border-b border-gray-800">
          <button
              onClick={() => setShowChat(false)}
              className="flex items-center mx-10 px-4 py-2 bg-[#1D1D1F] rounded-lg hover:bg-gray-700 transition-all duration-300 text-gray-300 group"
            >
              <ArrowLeft className="ml -2 group-hover:-translate-x-1 transition-transform" />
              Back to Home
            </button>
            <h1 className="text-xl font-bold ml-2 text-gray-300">SuperFloat </h1>
            
          </div>
          <div className="flex-1">
            <ChatInterface />
          </div>
        </div>
      ) : (
        <div className="max-w-6xl mx-auto px-4 py-16">
          {/* Hero Section */}
          <div className="text-center space-y-8 animate-fade-in">
            <h1 className="text-6xl font-bold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-transparent bg-clip-text animate-text">
              EmelinLabs presents
            </h1>
            <h2 className="text-8xl font-extrabold animate-float">
              Superfloat
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto animate-fade-in-up">
            A revolutionary quantization algorithm optimizing neural networks through custom precision formats. 
              Designed for edge computing with scalable precision and superior performance.
            </p>
            <button
              onClick={() => setShowChat(true)}
              className="group px-8 py-4 bg-indigo-600 rounded-full text-xl font-semibold hover:bg-indigo-700 transition-all duration-300 animate-bounce-slow"
            >
              Try Now 
              <ArrowRight className="inline ml-2 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* What We Do Section */}
          <div className="mt-32 space-y-8">
            <h3 className="text-4xl font-bold text-center">What We Do</h3>
            <div className="grid md:grid-cols-3 gap-8 mt-8">
              {[
                {
                    title: "Sign-Exponent Representation",
                    description: "Efficient bit allocation with 1 bit for sign and remaining for exponent, optimizing precision without mantissa."
                },
                {
                    title: "Clamping Range",
                    description: "Values clamped within [-1, 1] for activation and parameter stability, preventing gradient issues."
                },
                {
                    title: "Bit-width Flexibility",
                    description: "Scalable precision from 3-bit to 16-bit, balancing computation speed and accuracy."
                
                }
              ].map((feature, index) => (
                <div 
                  key={index}
                  className="p-6 bg-gray-800 rounded-xl hover:bg-gray-750 transition-all duration-300 transform hover:-translate-y-2"
                >
                  <h4 className="text-xl font-semibold mb-4">{feature.title}</h4>
                  <p className="text-gray-400">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* About Section */}
          <div className="mt-32 space-y-8">
            <h3 className="text-4xl font-bold text-center">About Us</h3>
            <p className="text-xl text-gray-400 text-center max-w-3xl mx-auto">
            SuperFloat implements custom quantization algorithms focusing on the Lottery Ticket Hypothesis (LTH) 
              and Weight and Activation SuperFloat Quantization (WASQ) techniques for optimizing neural networks 
              on edge devices.
            </p>
          </div>

          {/* Contact Section */}
          <div className="mt-16 flex justify-center space-x-6">
            <a
              href="https://github.com/aloshdenny/superfloat-accelerator"
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center px-6 py-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-all duration-300"
            >
              <Github className="mr-2" />
              <span>GitHub</span>
            </a>
            <a
              href="mailto:contact@ensdinlabs.com"
              className="group flex items-center px-6 py-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-all duration-300"
            >
              <Mail className="mr-2" />
              <span>Contact Us</span>
            </a>
          </div>
        </div>
      )}

      <style jsx global>{`
        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
          100% { transform: translateY(0px); }
        }

        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes fade-in-up {
          from { 
            opacity: 0;
            transform: translateY(20px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes text {
          0%, 100% {
            background-size: 200% 200%;
            background-position: left center;
          }
          50% {
            background-size: 200% 200%;
            background-position: right center;
          }
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }

        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }

        .animate-fade-in-up {
          animation: fade-in-up 1s ease-out;
        }

        .animate-text {
          animation: text 5s ease infinite;
        }

        .animate-bounce-slow {
          animation: bounce 2s infinite;
        }
      `}</style>
    </div>
  );
};

export default LandingPage;