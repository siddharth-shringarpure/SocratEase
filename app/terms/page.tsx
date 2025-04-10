"use client";

import {
  motion,
  AnimatePresence,
  useInView,
  useScroll,
  useSpring,
  useAnimationFrame,
} from "framer-motion";
import Head from "next/head";
import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { ArrowUp, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import React from "react";
import * as Accordion from "@radix-ui/react-accordion";

/**
 * Terms and Conditions page component
 * @returns {JSX.Element} The terms and conditions page
 */
export default function TermsAndConditions(): JSX.Element {
  // State to track active section for table of contents
  const [activeSection, setActiveSection] = useState<string>("intro");

  // State to track if mouse is near sidebar for focus effect
  const [isMouseNearSidebar, setIsMouseNearSidebar] = useState<boolean>(false);

  // State to track if page has loaded
  const [pageLoaded, setPageLoaded] = useState<boolean>(false);

  // Refs for each section
  const sectionRefs = {
    intro: useRef<HTMLElement>(null),
    services: useRef<HTMLElement>(null),
    account: useRef<HTMLElement>(null),
    content: useRef<HTMLElement>(null),
    intellectual: useRef<HTMLElement>(null),
    privacy: useRef<HTMLElement>(null),
    liability: useRef<HTMLElement>(null),
    changes: useRef<HTMLElement>(null),
    contact: useRef<HTMLElement>(null),
  };

  // Main content ref for handling focus/blur effect
  const contentRef = useRef<HTMLDivElement>(null);

  // Remove isScrolling state as it's not needed
  const [targetScrollY, setTargetScrollY] = useState<number | null>(null);

  // Create a spring animation for smooth scrolling
  const springConfig = { stiffness: 80, damping: 20, mass: 1 };
  const scrollY = useSpring(0, springConfig);

  // Use animation frame to update scroll position
  useAnimationFrame(() => {
    if (targetScrollY !== null) {
      const currentScroll = window.scrollY;
      const diff = Math.abs(targetScrollY - currentScroll);

      if (diff < 1) {
        setTargetScrollY(null);
        return;
      }

      // Calculate a step towards the target with spring-like easing
      const step = (targetScrollY - currentScroll) * 0.1;
      window.scrollTo(0, currentScroll + step);
    }
  });

  // Update active section based on scroll position
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 150;

      // Find the section that is currently in view
      let currentSection = activeSection;
      for (const sectionId of Object.keys(sectionRefs)) {
        const element =
          sectionRefs[sectionId as keyof typeof sectionRefs].current;
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (
            scrollPosition >= offsetTop &&
            scrollPosition < offsetTop + offsetHeight
          ) {
            currentSection = sectionId;
            break;
          }
        }
      }

      if (activeSection !== currentSection) {
        setActiveSection(currentSection);
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [activeSection, sectionRefs]);

  const currentDate = new Date().toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  const sections = [
    { id: "intro", label: "Introduction" },
    { id: "services", label: "Our Services" },
    { id: "account", label: "Using the Platform" },
    { id: "content", label: "User Content" },
    { id: "intellectual", label: "Intellectual Property" },
    { id: "privacy", label: "Privacy Policy" },
    { id: "liability", label: "Liability & Warranty" },
    { id: "changes", label: "Changes to Terms" },
    { id: "contact", label: "Contact Us" },
  ];

  const scrollToSection = (sectionId: string) => {
    const element = sectionRefs[sectionId as keyof typeof sectionRefs].current;
    if (element) {
      const headerOffsetTop = -72;
      const targetOffsetTop = element.offsetTop;
      const scrollPosition = targetOffsetTop - headerOffsetTop;

      setTargetScrollY(scrollPosition);
      scrollY.set(scrollPosition);
      setActiveSection(sectionId);
    }
  };

  const scrollToTop = () => {
    const currentScroll = window.scrollY;
    setTargetScrollY(0);
    scrollY.set(currentScroll);
  };

  // Animation variants for sidebar elements
  const sidebarContainerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.5,
      },
    },
  };

  const sidebarItemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5,
        ease: [0.25, 0.1, 0.25, 1.0],
      },
    },
  };

  // Animation variants for staggered animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.7,
        ease: [0.25, 0.1, 0.25, 1.0],
      },
    },
  };

  // Effect to set page as loaded after a short delay
  useEffect(() => {
    const timer = setTimeout(() => {
      setPageLoaded(true);
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  // Reusable Radix Accordion Item Component
  const AccordionItem = React.forwardRef<
    HTMLDivElement,
    Accordion.AccordionItemProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Item
      className={`my-6 border border-border rounded-md overflow-hidden ${className}`}
      {...props}
      ref={forwardedRef}
    >
      {children}
    </Accordion.Item>
  ));
  AccordionItem.displayName = "AccordionItem";

  const AccordionTrigger = React.forwardRef<
    HTMLButtonElement,
    Accordion.AccordionTriggerProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Header className="flex">
      <Accordion.Trigger
        className={`group w-full p-4 flex justify-between items-center text-left bg-secondary/5 
        hover:bg-secondary/10 transition-colors data-[state=open]:bg-secondary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className}`}
        {...props}
        ref={forwardedRef}
      >
        <span className="font-semibold">{children}</span>
        <ChevronDown
          size={18}
          className="transition-transform duration-300 ease-[cubic-bezier(0.87,_0,_0.13,_1)] group-data-[state=open]:rotate-180"
          aria-hidden
        />
      </Accordion.Trigger>
    </Accordion.Header>
  ));
  AccordionTrigger.displayName = "AccordionTrigger";

  const AccordionContent = React.forwardRef<
    HTMLDivElement,
    Accordion.AccordionContentProps & {
      children: React.ReactNode;
      className?: string;
    }
  >(({ children, className, ...props }, forwardedRef) => (
    <Accordion.Content
      className={`data-[state=open]:animate-slideDown data-[state=closed]:animate-slideUp overflow-hidden relative z-30 ${className}`}
      {...props}
      ref={forwardedRef}
    >
      <div className="p-6 bg-background text-foreground/80 text-sm">
        {children}
      </div>
    </Accordion.Content>
  ));
  AccordionContent.displayName = "AccordionContent";

  // Section with blur effect that fades as it comes into view
  const BlurFadeSection = ({ children }: { children: React.ReactNode }) => {
    const ref = useRef(null);
    const isInView = useInView(ref, {
      margin: "-10% 0px -10% 0px",
      amount: 0.3,
    });

    return (
      <motion.div
        ref={ref}
        animate={{
          filter: isInView ? "blur(0px)" : "blur(5px)",
          opacity: isInView ? 1 : 0.7,
        }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        {children}
      </motion.div>
    );
  };

  // Terms Section component
  const TermsSection = ({
    id,
    title,
    children,
  }: {
    id: string;
    title: string;
    children: React.ReactNode;
  }) => {
    const ref = sectionRefs[id as keyof typeof sectionRefs];

    return (
      <section
        id={id}
        ref={ref}
        className="mb-36 last:mb-0 scroll-mt-24 relative"
      >
        {/* Sticky header implementation */}
        <div className="sticky top-[72px] z-20">
          {/* Shield div using padding instead of negative positioning */}
          <div className="absolute inset-x-0 top-0 bg-background z-10">
            <div className="pt-16 pb-4" />
          </div>

          {/* Header content with higher z-index */}
          <div className="bg-background/95 backdrop-blur-sm relative z-20">
            <div className="py-6">
              <h2 className="text-3xl font-semibold">{title}</h2>
            </div>
            {/* Gradient fade for smooth transition */}
            <div className="absolute -bottom-6 left-0 right-0 h-6 bg-gradient-to-b from-background/95 to-transparent pointer-events-none" />
          </div>
        </div>

        {/* Content container with consistent spacing */}
        <div className="pt-24 pb-16 space-y-10 relative z-10">
          <BlurFadeSection>{children}</BlurFadeSection>
        </div>
      </section>
    );
  };

  return (
    <>
      <Head>
        <title>Terms of Service - SocratEase</title>
        <meta
          name="description"
          content="SocratEase Terms of Service and Legal Information"
        />
      </Head>

      <div className="bg-background min-h-screen pb-32">
        {/* Fixed back to top button */}
        <motion.button
          onClick={scrollToTop}
          className="fixed bottom-8 right-8 z-50 p-3 bg-background/80 backdrop-blur-sm border border-border rounded-full shadow-lg hover:bg-background/95 transition-all group"
          aria-label="Back to top"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: pageLoaded ? 1 : 0, scale: pageLoaded ? 1 : 0.8 }}
          whileHover={{ scale: 1.1 }}
          transition={{ duration: 0.2 }}
        >
          <ArrowUp className="h-5 w-5 text-foreground group-hover:-translate-y-0.5 transition-transform" />
        </motion.button>

        <div className="container mx-auto px-6 pt-32 pb-16 relative">
          {/* Blur Overlay - Positioned before content */}
          <motion.div
            className="fixed inset-0 pointer-events-none z-10"
            initial={{ opacity: 0 }}
            animate={{
              opacity: isMouseNearSidebar ? 1 : 0,
            }}
            transition={{ duration: 0.4 }}
            style={{
              backdropFilter: isMouseNearSidebar ? "blur(4px)" : "blur(0px)",
              background: isMouseNearSidebar
                ? "rgba(0, 0, 0, 0.2)"
                : "rgba(0, 0, 0, 0)",
              transition: "backdrop-filter 0.4s ease, background 0.4s ease",
            }}
          />

          {/* Sidebar navigation - Further away from content */}
          <div
            className="hidden lg:block fixed left-10 xl:left-20 2xl:left-40 top-1/2 -translate-y-1/2 z-30"
            onMouseEnter={() => setIsMouseNearSidebar(true)}
            onMouseLeave={() => setIsMouseNearSidebar(false)}
          >
            <motion.div
              className="flex flex-col space-y-6 items-center relative z-10 p-4 -m-4"
              variants={sidebarContainerVariants}
              initial="hidden"
              animate="visible"
              whileHover={{
                opacity: isMouseNearSidebar ? 1 : 0.7,
                scale: isMouseNearSidebar ? 1.05 : 1,
              }}
              transition={{ duration: 0.3 }}
            >
              {sections.map((section) => (
                <motion.button
                  key={section.id}
                  onClick={() => scrollToSection(section.id)}
                  className="group relative flex items-center w-full"
                  aria-label={section.label}
                  variants={sidebarItemVariants}
                >
                  <motion.div
                    className={`w-3 h-3 rounded-full transition-colors duration-300 ${
                      activeSection === section.id
                        ? "bg-primary"
                        : "bg-secondary/40 group-hover:bg-secondary/70"
                    }`}
                    whileHover={{ scale: 1.2 }}
                    animate={{
                      scale: activeSection === section.id ? 1.2 : 1,
                      boxShadow:
                        activeSection === section.id
                          ? "0 0 8px rgba(var(--primary), 0.5)"
                          : "none",
                    }}
                  ></motion.div>
                  <motion.span
                    className={`absolute left-6 py-2 px-4 text-xs rounded whitespace-nowrap bg-background/80 backdrop-blur-sm min-h-[32px] flex items-center ${
                      activeSection === section.id
                        ? "text-primary"
                        : "text-foreground/70"
                    }`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{
                      opacity:
                        isMouseNearSidebar || activeSection === section.id
                          ? 1
                          : 0,
                      x:
                        isMouseNearSidebar || activeSection === section.id
                          ? 0
                          : -10,
                    }}
                    transition={{ duration: 0.2 }}
                  >
                    {section.label}
                  </motion.span>
                </motion.button>
              ))}
            </motion.div>
          </div>

          {/* Main content area - Centered with proper max width */}
          <div className="max-w-3xl mx-auto w-full relative">
            {/* Actual Content Container */}
            <div ref={contentRef} className="relative z-10">
              {/* Title section */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="text-center mb-40"
              >
                <motion.h1
                  className="text-4xl md:text-5xl font-bold text-center mb-8"
                  variants={itemVariants}
                >
                  Terms of Service
                </motion.h1>
                <motion.p
                  className="text-foreground/80 text-center max-w-3xl mx-auto mb-12"
                  variants={itemVariants}
                >
                  Please read these terms carefully before using the SocratEase
                  platform. By accessing or using our services, you agree to be
                  bound by these Terms of Service.
                </motion.p>
                <p className="text-muted-foreground">
                  Last updated: {currentDate}
                </p>
              </motion.div>

              {/* Content Sections Container - Use Radix Accordion Root here */}
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate={pageLoaded ? "visible" : "hidden"}
              >
                <Accordion.Root
                  type="multiple" // Allow multiple items open at once
                  className="space-y-0" // TermsSection handles margin
                >
                  <motion.div variants={itemVariants}>
                    <TermsSection id="intro" title="Introduction">
                      {/* Introduction to Terms of Service */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          SocratEase is an AI-powered platform that helps you
                          improve your communication skills by transforming your
                          speech videos into insightful feedback.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          Throughout these terms, "us," "we," "our," and
                          "SocratEase" refer to SocratEase and its associated
                          services, and "you" refers to the individual or entity
                          accessing and using the Service. By using SocratEase,
                          you affirm that you are at least 18 years old or have
                          obtained valid consent from a parent or guardian.
                        </p>

                        <p className="text-foreground/80 leading-relaxed">
                          If you do not agree with these terms, you may not use
                          our services. By accessing or using SocratEase, you
                          represent and warrant that you have the right,
                          authority, and capacity to enter into this agreement
                          and to abide by all of the terms and conditions.
                        </p>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">⚠️</span>
                            <h3 className="text-xl font-semibold">
                              Prohibited Activities
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            You are prohibited from using SocratEase or its
                            content:
                          </p>
                          <ul className="list-disc pl-8 space-y-3 text-foreground/80 mt-4">
                            <li>
                              Use our services for any illegal purpose or in
                              violation of any laws
                            </li>
                            <li>
                              To violate any international, federal, provincial
                              or state regulations, rules, laws, or local
                              ordinances
                            </li>
                            <li>
                              To infringe upon or violate our intellectual
                              property rights or the intellectual property
                              rights of others
                            </li>
                            <li>
                              To harass, abuse, insult, harm, defame, slander,
                              disparage, intimidate, or discriminate based on
                              gender, sexual orientation, religion, ethnicity,
                              race, age, national origin, or disability
                            </li>
                            <li>To submit false or misleading information</li>
                            <li>
                              To upload or transmit viruses or any other type of
                              malicious code
                            </li>
                            <li>
                              To interfere with or circumvent the security
                              features of the Service
                            </li>
                          </ul>
                        </div>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="services" title="Our Services">
                      {/* Services description */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          SocratEase provides AI-powered speech practice and
                          feedback services. Our platform allows you to practise
                          presentations, interviews, and speeches with detailed
                          feedback and analysis.
                        </p>
                      </div>

                      <div className="mt-16 space-y-10">
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🎙️</span>
                            <h3 className="text-xl font-semibold">
                              Speech Practice
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Our platform enables you to record practice
                            sessions, whether for presentations, interviews, or
                            other speaking engagements. You can practise in a
                            safe environment before your real-world performance.
                          </p>
                        </div>

                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">📊</span>
                            <h3 className="text-xl font-semibold">
                              AI Analysis
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Our AI technology analyses your speech patterns,
                            delivery, and content, providing you with detailed
                            feedback to help you improve. This includes insights
                            on pacing, clarity, vocabulary, and more.
                          </p>
                        </div>

                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">📝</span>
                            <h3 className="text-xl font-semibold">
                              Personalised Guidance
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Based on your practice sessions, we provide tailored
                            recommendations to enhance your speaking skills. Our
                            service aims to help you become a more confident and
                            effective communicator.
                          </p>
                        </div>

                        <AccordionItem value="service-limitations">
                          <AccordionTrigger>
                            Service Limitations and Availability
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="space-y-4">
                              <p>
                                SocratEase services are provided on an "as is"
                                and "as available" basis. We may experience
                                interruptions, delays, or errors in service at
                                any time without notice.
                              </p>
                              <p>
                                We reserve the right to modify, suspend, or
                                discontinue any part of our services at any
                                time. We will make reasonable efforts to notify
                                users of significant changes that affect their
                                use of our platform.
                              </p>
                              <p>
                                Whilst we strive to provide accurate and helpful
                                feedback, our AI analysis is not perfect and
                                should be considered as guidance rather than
                                absolute assessment. The quality of feedback
                                depends on various factors including audio
                                quality, speaking clarity, and technical
                                conditions.
                              </p>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="account" title="Using the Platform">
                      {/* Platform usage terms */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          SocratEase is designed to be simple to use without
                          requiring account creation. However, there are
                          important guidelines about how you can use our
                          platform.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          When using SocratEase, you agree not to use our
                          services for any illegal, harmful, or offensive
                          purposes. This includes, but is not limited to:
                        </p>

                        <ul className="list-disc pl-8 space-y-3 text-foreground/80">
                          <li>
                            Using our platform to practise or create content
                            that promotes hate speech, discrimination,
                            harassment, or violence
                          </li>
                          <li>
                            Attempting to interfere with or disrupt the
                            operation of our services or servers
                          </li>
                          <li>
                            Using automated systems or software to extract data
                            from our platform (scraping)
                          </li>
                          <li>
                            Attempting to access parts of the service that you
                            don't have permission to access
                          </li>
                          <li>
                            Using our services in a way that could disable,
                            overburden, or impair the functionality of the
                            platform
                          </li>
                        </ul>

                        <div className="bg-primary/5 p-6 rounded-lg border border-primary/20 shadow-sm mt-8">
                          <div className="flex items-center mb-3">
                            <span className="text-primary font-semibold">
                              Non-Discrimination and Proper Usage:
                            </span>
                          </div>
                          <p className="text-foreground/90 leading-relaxed">
                            SocratEase must not be used for making employment
                            decisions, determining access to educational
                            opportunities, evaluating eligibility for financial
                            services, or any similar high-stakes decision-making
                            that could impact an individual's rights,
                            opportunities, or access to resources. Our platform
                            is designed as a learning and practice tool only,
                            and should never be used to discriminate against
                            individuals or groups.
                          </p>
                        </div>

                        <p className="text-foreground/80 leading-relaxed mt-6">
                          We reserve the right to terminate or restrict your
                          access to our services if you violate these terms or
                          engage in any activity that we deem harmful to other
                          users, third parties, or the operation of our
                          services.
                        </p>
                      </div>

                      <AccordionItem value="age-requirements" className="mt-10">
                        <AccordionTrigger>
                          Age Requirements and Parental Consent
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="space-y-4">
                            <p>
                              You must be at least 16 years old to use
                              SocratEase. By accessing or using our services,
                              you confirm that you meet this age requirement.
                            </p>
                            <p>
                              If we discover that a person under 16 is using our
                              services without parental consent, we may
                              terminate their access to the platform. We
                              encourage parents and guardians to monitor their
                              children's internet usage and to help enforce
                              these Terms.
                            </p>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="content" title="User Content">
                      {/* User content terms */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          When you use SocratEase, you may submit content such
                          as speech recordings, presentation materials, and
                          practice sessions. Here's how we handle this content.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <div>
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🔐</span>
                            <h3 className="text-xl font-semibold">
                              Ownership of Your Content
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            You retain ownership of any content you submit to
                            SocratEase. However, by uploading content to our
                            platform, you grant us a limited licence to use,
                            process, and store your content solely for the
                            purpose of providing our services to you.
                          </p>
                        </div>

                        <div className="mt-10">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🧠</span>
                            <h3 className="text-xl font-semibold">
                              AI Processing and Data Usage
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed mb-4">
                            We use AI to analyse your practice sessions and
                            provide feedback. Your content is processed
                            automatically to generate this feedback.
                          </p>
                          <div className="bg-primary/5 p-6 rounded-lg border border-primary/20 shadow-sm">
                            <div className="flex items-center mb-3">
                              <span className="text-primary font-semibold">
                                Important:
                              </span>
                            </div>
                            <p className="text-foreground/90 leading-relaxed">
                              As stated in our Privacy Policy, we{" "}
                              <strong>do not</strong> use your audio, video, or
                              practice content to train our AI models. Your
                              content is only used to provide immediate feedback
                              to you, and is then deleted after processing.
                            </p>
                          </div>
                        </div>

                        <div className="mt-10">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">⚠️</span>
                            <h3 className="text-xl font-semibold">
                              Content Restrictions
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            You are solely responsible for the content you
                            submit. You agree not to submit content that:
                          </p>
                          <ul className="list-disc pl-8 space-y-3 text-foreground/80 mt-4">
                            <li>
                              Infringes on intellectual property rights of
                              others
                            </li>
                            <li>
                              Contains illegal, offensive, harmful, or
                              inappropriate material
                            </li>
                            <li>
                              Contains malware, viruses, or other malicious code
                            </li>
                            <li>
                              Violates the privacy or publicity rights of others
                            </li>
                            <li>
                              Misrepresents your identity or affiliation with
                              any person or organisation
                            </li>
                          </ul>
                        </div>

                        <AccordionItem value="content-removal" className="mt-6">
                          <AccordionTrigger>
                            Content Removal and Reporting
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="space-y-4">
                              <p>
                                We reserve the right to remove any content that
                                violates these Terms or that we find
                                objectionable for any reason, without prior
                                notice.
                              </p>
                              <p>
                                If you encounter content on our platform that
                                you believe violates these Terms or is otherwise
                                inappropriate, please contact us immediately at
                                hello.socratease@proton.me.
                              </p>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection
                      id="intellectual"
                      title="Intellectual Property"
                    >
                      {/* Intellectual property section */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          SocratEase and its original content, features, and
                          functionality are owned by us and are protected by
                          international copyright, trademark, and other
                          intellectual property laws.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          Our trademarks and trade dress may not be used in
                          connection with any product or service without our
                          prior written consent. The design, layout, look,
                          appearance, and graphics of our website and platform
                          are protected works, and copying or imitating them is
                          strictly prohibited.
                        </p>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">📜</span>
                            <h3 className="text-xl font-semibold">
                              Permitted Use
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            We grant you a limited, non-exclusive,
                            non-transferable, and revocable licence to use
                            SocratEase for your personal, non-commercial use in
                            accordance with these Terms. This licence does not
                            include:
                          </p>
                          <ul className="list-disc pl-8 space-y-3 text-foreground/80 mt-4">
                            <li>
                              Modifying or creating derivative works based on
                              our platform or its contents
                            </li>
                            <li>
                              Using any data mining, robots, or similar data
                              gathering methods
                            </li>
                            <li>
                              Downloading or copying account information for the
                              benefit of another party
                            </li>
                            <li>
                              Using our services beyond their intended purpose
                            </li>
                          </ul>
                        </div>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🚫</span>
                            <h3 className="text-xl font-semibold">
                              Feedback and Submissions
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            Any feedback, suggestions, ideas, or other
                            information you provide to us regarding our services
                            may be used by us without restriction or
                            compensation to you. By submitting such information,
                            you grant us all rights to use and incorporate it
                            into our services.
                          </p>
                        </div>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="privacy" title="Privacy Policy">
                      {/* Privacy policy section */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          Your privacy is important to us. Our Privacy Policy
                          explains how we collect, use, and protect your
                          personal information.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          By using SocratEase, you consent to the data practices
                          described in our
                          <Link
                            href="/privacy"
                            className="text-primary hover:underline mx-1"
                          >
                            Privacy Policy
                          </Link>
                          which is incorporated into these Terms of Service. We
                          encourage you to read it carefully.
                        </p>

                        <div className="bg-primary/5 p-6 rounded-lg border border-primary/20 shadow-sm mt-8">
                          <div className="flex items-center mb-3">
                            <span className="text-primary font-semibold">
                              Key Privacy Points:
                            </span>
                          </div>
                          <ul className="list-disc pl-6 space-y-3 text-foreground/90">
                            <li>
                              We only collect data necessary to provide our
                              services
                            </li>
                            <li>
                              Your practice recordings are temporarily processed
                              and then deleted
                            </li>
                            <li>
                              We do not use your content to train our AI models
                            </li>
                            <li>We do not sell your personal information</li>
                            <li>
                              Your settings are stored locally on your device
                            </li>
                          </ul>
                        </div>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="liability" title="Liability & Warranty">
                      {/* Liability section */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          Our services are provided "as is" without any warranty
                          or condition, express, implied or statutory.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          SocratEase and its developers, owners, and affiliates
                          make no warranties or representations about the
                          accuracy, reliability, completeness, or timeliness of
                          the content, services, software, text, graphics, and
                          links used on our platform.
                        </p>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">⚖️</span>
                            <h3 className="text-xl font-semibold">
                              Limitation of Liability
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            To the maximum extent permitted by law, SocratEase,
                            its affiliates, and their respective directors,
                            employees, agents, and licensors shall not be liable
                            for any indirect, incidental, special,
                            consequential, or punitive damages arising out of or
                            relating to your use of our services.
                          </p>
                          <p className="text-foreground/80 leading-relaxed mt-4">
                            This includes, but is not limited to, any errors or
                            omissions in any content, loss of data, or any other
                            loss or damage of any kind incurred as a result of
                            your use of the service, even if advised of their
                            possibility.
                          </p>
                        </div>

                        <AccordionItem value="disclaimer" className="mt-6">
                          <AccordionTrigger>
                            Service Disclaimer and AI Limitations
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="space-y-4">
                              <p>
                                SocratEase uses artificial intelligence to
                                provide feedback and analysis. Whilst we strive
                                for high quality and accuracy, AI technology has
                                inherent limitations and may not always provide
                                perfect or complete feedback.
                              </p>
                              <p>
                                The feedback provided by our platform should be
                                considered as guidance rather than definitive
                                assessment. We do not guarantee that our
                                services will improve your speaking skills or
                                lead to specific outcomes in real-world
                                scenarios.
                              </p>
                              <p>
                                <strong>Important:</strong> SocratEase provides
                                feedback for practice and learning purposes
                                only. Our services must not be used for making
                                employment decisions, determining educational
                                opportunities, or any other official assessments
                                that could impact an individual's rights or
                                access to opportunities. Using SocratEase for
                                such purposes could lead to unfair or
                                discriminatory outcomes and is strictly
                                prohibited.
                              </p>
                              <p>
                                You acknowledge that AI analysis may contain
                                errors, inaccuracies, or limitations, and you
                                agree to use your own judgement when applying
                                the feedback provided through our services.
                              </p>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="changes" title="Changes to Terms">
                      {/* Changes to terms section */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <p className="text-xl leading-relaxed">
                          We reserve the right to modify these Terms at any
                          time. Changes will be effective immediately upon
                          posting.
                        </p>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          It is your responsibility to review these Terms
                          periodically to stay informed of updates. Your
                          continued use of SocratEase after the posting of
                          revised Terms means that you accept and agree to the
                          changes.
                        </p>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">📢</span>
                            <h3 className="text-xl font-semibold">
                              Notification of Changes
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            For significant changes to these Terms, we will make
                            reasonable efforts to provide notice, such as a
                            prominent notice on our website or an email
                            notification if you have provided contact
                            information. What constitutes a significant change
                            will be determined at our sole discretion.
                          </p>
                        </div>

                        <p className="text-foreground/80 leading-relaxed mt-8">
                          The latest version of these Terms will always be
                          available on this page, with the "Last updated" date
                          at the top. We encourage you to check this page
                          regularly to stay informed about our legal terms.
                        </p>

                        <div className="mt-8">
                          <div className="flex items-center mb-4">
                            <span className="text-3xl mr-3">🔄</span>
                            <h3 className="text-xl font-semibold">
                              Service Modifications
                            </h3>
                          </div>
                          <p className="text-foreground/80 leading-relaxed">
                            We reserve the right to update, modify, or replace
                            any part of the Service without prior notice. We may
                            also change or discontinue specific features or
                            functionality of the Service. We are not liable to
                            you or any third party for any modification,
                            suspension, or discontinuance of the Service.
                          </p>
                        </div>
                      </div>
                    </TermsSection>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <TermsSection id="contact" title="Contact Us">
                      {/* Contact section */}
                      <div className="bg-gradient-to-br from-secondary/5 to-primary/5 p-8 rounded-lg shadow-sm">
                        <div className="flex flex-col md:flex-row md:items-center justify-between">
                          <div>
                            <p className="text-xl mb-4">
                              Questions about these Terms?
                            </p>
                            <p className="text-foreground/80">
                              <strong>Email:</strong> hello.socratease@proton.me
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="mt-10 space-y-6">
                        <p className="text-foreground/80 leading-relaxed">
                          If you have any questions or concerns about these
                          Terms of Service or our platform, please contact us.
                          We are committed to addressing any issues and ensuring
                          a positive user experience.
                        </p>

                        <div className="mt-6">
                          <p className="text-foreground/80 leading-relaxed">
                            When contacting us regarding these Terms, please
                            include:
                          </p>
                          <ul className="list-disc pl-8 space-y-3 text-foreground/80 mt-4">
                            <li>
                              A clear description of your concern or question
                            </li>
                            <li>
                              Any relevant details about your use of our
                              platform
                            </li>
                            <li>
                              Your suggestions, if any, for resolving the issue
                            </li>
                          </ul>
                        </div>

                        <p className="text-foreground/80 leading-relaxed mt-6">
                          We will make reasonable efforts to respond to all
                          enquiries in a timely manner.
                        </p>

                        <div className="bg-primary/5 p-6 rounded-lg border border-primary/20 shadow-sm mt-8">
                          <div className="flex items-center mb-3">
                            <span className="text-primary font-semibold">
                              Governing Law
                            </span>
                          </div>
                          <p className="text-foreground/90 leading-relaxed">
                            These Terms shall be governed and construed in
                            accordance with the laws applicable in your
                            jurisdiction, without regard to its conflict of law
                            provisions. Our failure to enforce any right or
                            provision of these Terms will not be considered a
                            waiver of those rights.
                          </p>
                        </div>
                      </div>
                    </TermsSection>
                  </motion.div>

                  {/* Final statement */}
                  <motion.div
                    variants={itemVariants}
                    className="border-t border-border pt-8 mt-16 text-center"
                  >
                    <p className="text-muted-foreground">
                      By using SocratEase, you acknowledge that you have read,
                      understood, and agree to be bound by these Terms of
                      Service. If you disagree with any part of these terms, you
                      must discontinue use of our service immediately.
                    </p>
                  </motion.div>
                </Accordion.Root>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
